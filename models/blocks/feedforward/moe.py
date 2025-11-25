# FILE: models/blocks/feedforward/moe.py
"""
[v3.2 - Stable Aux-free Logic]
- 优化 Aux-free 负载均衡：增加 Bias 范围钳制 (Clamp) 以防数值震荡。
- 保持 CPU 友好的循环实现，但增加 DDP 警告。
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
        # 注意：在 DDP 模式下，这种条件执行路径（Conditional Computation）可能会导致
        # "unused parameters" 错误。
        # 必须在 DDP 初始化时设置 `find_unused_parameters=True`。
        final_output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 0, 1)

        # [统计负载]
        # 注意：在推理时不需要更新 bias，只在训练时统计
        if self.training and self.use_aux_free_lb:
            # expert_mask shape: (NumExperts, TotalTokens, TopK)
            # Sum over tokens and topk -> (NumExperts)
            # 使用 detach 避免影响梯度图
            current_counts = expert_mask.sum(dim=(1, 2)).detach().float()
            # 简单的覆盖更新，由 Trainer 定期调用 update_bias 处理
            self.last_expert_counts = current_counts

        for expert_idx in range(self.num_experts):
            idx_in_topk = torch.where(expert_mask[expert_idx] > 0)

            # 如果没有任何 token 选择这个专家，直接跳过计算
            # 这是 CPU 推理加速的关键
            if idx_in_topk[0].numel() == 0:
                continue

            token_indices = idx_in_topk[0]

            # 只计算被选中的 token
            expert_output = self.experts[expert_idx](x_flat[token_indices])

            # 加权累加
            weights = routing_weights[token_indices, idx_in_topk[1]].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_output * weights)

        if self.num_shared_experts > 0:
            final_output = final_output + shared_output

        return final_output.view(batch_size, seq_len, dim)

    def update_bias(self, lr: float = 0.05):
        """
        [DeepSeek-V3] Aux-free Load Balancing 更新逻辑。

        [改进]: 增加 Clamp 防止 bias 无限制增长导致的“死专家”问题。
        """
        if not self.use_aux_free_lb:
            return

        counts = self.last_expert_counts
        # 加上 1.0 避免全 0 的情况
        mean_count = counts.mean() + 1e-6

        # 如果某个专家的负载远超平均值，bias 应该降低
        # 如果某个专家没人选，bias 应该升高
        # sign(count - mean) > 0 => count > mean => bias decreases

        error = counts - mean_count
        update_step = lr * torch.sign(error)

        self.expert_bias.data -= update_step

        # [核心修复] 钳制 bias 范围，防止极端值
        # 限制在 [-10, 10] 之间，这对于 Logits 来说已经足够大了
        self.expert_bias.data.clamp_(min=-10.0, max=10.0)

        # 归零统计量
        self.last_expert_counts.zero_()

# END OF FILE: models/blocks/feedforward/moe.py