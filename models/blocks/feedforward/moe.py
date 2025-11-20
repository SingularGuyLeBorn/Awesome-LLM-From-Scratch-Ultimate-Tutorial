# FILE: models/blocks/feedforward/moe.py
"""
[v3.1 - Aux-free Update Logic]
- 增加 last_expert_counts 缓冲区，用于记录专家负载。
- 实现 update_bias 方法，根据负载偏差动态调整 expert_bias。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .feedforward import FeedForward


class MoELayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int, num_experts_per_tok: int,
                 multiple_of: int = 256, num_shared_experts: int = 0, use_aux_free_lb: bool = False):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.use_aux_free_lb = use_aux_free_lb

        self.router = nn.Linear(dim, num_experts, bias=False)

        if self.use_aux_free_lb:
            # 动态 Bias，不通过梯度更新，而是通过统计规律更新
            self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of) for _ in range(num_experts)
        ])

        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(dim, hidden_dim, multiple_of) for _ in range(num_shared_experts)
            ])

        # 统计 buffer
        self.register_buffer('last_expert_counts', torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # --- Shared Experts Path ---
        shared_output = 0
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output += expert(x_flat)

        # --- Routed Experts Path ---
        router_logits = self.router(x_flat)

        if self.use_aux_free_lb:
            router_logits = router_logits + self.expert_bias

        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)

        # --- CPU Friendly Loop ---
        final_output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 0, 1)

        # [统计负载]
        # 注意：在推理时不需要更新 bias，只在训练时统计
        if self.training and self.use_aux_free_lb:
            # expert_mask shape: (NumExperts, TotalTokens, TopK)
            # Sum over tokens and topk -> (NumExperts)
            # 使用 detach 避免影响梯度图
            current_counts = expert_mask.sum(dim=(1, 2)).detach().float()
            # 简单的移动平均，或者直接覆盖（由 Trainer 决定如何使用）
            self.last_expert_counts = current_counts

        for expert_idx in range(self.num_experts):
            idx_in_topk = torch.where(expert_mask[expert_idx] > 0)
            if idx_in_topk[0].numel() == 0:
                continue

            token_indices = idx_in_topk[0]
            expert_output = self.experts[expert_idx](x_flat[token_indices])
            weights = routing_weights[token_indices, idx_in_topk[1]].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_output * weights)

        if self.num_shared_experts > 0:
            final_output = final_output + shared_output

        return final_output.view(batch_size, seq_len, dim)

    def update_bias(self, lr: float = 0.05):
        """
        [DeepSeek-V3] Aux-free Load Balancing 更新逻辑。

        逻辑：
        如果一个专家被选中的次数 (last_expert_counts) 超过了平均值，说明它"过热"。
        我们需要降低它的 Bias，让它在下一次更难被选中。
        反之，如果专家"过冷"，增加 Bias。

        公式： bias -= lr * sign(count - mean_count)
        """
        if not self.use_aux_free_lb:
            return

        counts = self.last_expert_counts
        target = counts.mean()  # 理想情况下是均匀分布

        # 计算偏差
        error = counts - target

        # 更新 bias (In-place)
        # 注意：这是一个纯数值操作，不涉及 Autograd
        update_step = lr * torch.sign(error)
        self.expert_bias.data -= update_step

        # 归零统计量 (可选，取决于是否累计)
        self.last_expert_counts.zero_()

# END OF FILE: models/blocks/feedforward/moe.py