# FILE: models/blocks/feedforward/moe.py
"""
[新增] 稀疏混合专家 (Sparse Mixture of Experts, MoE) 层的实现。

核心逻辑:
1. Router (Gate): 一个线性层，预测每个 Token 应该分配给哪些专家。
2. Top-K Selection: 选择得分最高的 K 个专家。
3. Dispersion & Aggregation:
   - 在纯PyTorch/CPU实现中，为了代码清晰和易于调试，我们采用循环或掩码的方式。
   - 对于高性能实现，通常需要 optimized kernels。此处我们采用"加权求和"的逻辑，
     即 output = sum( prob[i] * expert[i](x) )。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .feedforward import FeedForward


class MoELayer(nn.Module):
    """
    Sparse MoE Layer using Top-K Gating.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, num_experts_per_tok: int, multiple_of: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # 1. 路由/门控网络
        self.router = nn.Linear(dim, num_experts, bias=False)

        # 2. 专家列表 (ModuleList)
        # 注意：每个专家都是一个独立的 SwiGLU FFN
        # 优化提示：在 MoE 中，专家的 hidden_dim 通常比 Dense 模型小，以控制总参数量
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape

        # 将 batch 和 seq_len 展平，方便处理
        # x_flat: (total_tokens, dim)
        x_flat = x.view(-1, dim)

        # 1. 计算路由 logits: (total_tokens, num_experts)
        router_logits = self.router(x_flat)

        # 2. Top-K 选专家
        # routing_weights: (total_tokens, k), selected_experts: (total_tokens, k)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)

        # 3. 对权重进行 Softmax 归一化
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)

        # 4. 计算专家输出
        # --- CPU/Python 友好实现 (Naive Loop) ---
        # 这种方法虽然不如 CUDA kernel 快，但在 Python 层面最清晰，且支持动态图
        # 我们初始化输出张量
        final_output = torch.zeros_like(x_flat)

        # 为了利用 batch 计算，我们遍历所有专家
        # 如果某个专家被某些 token 选中了，我们就计算它

        # 创建一个 expert_mask: (num_experts, total_tokens)
        # expert_mask[e, t] = 1 如果 token t 选中了专家 e
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 0, 1)

        for expert_idx in range(self.num_experts):
            # 找出选中当前专家 expert_idx 的 token 索引
            # idx_in_topk: 它是第几个被选中的 (1st choice or 2nd choice?)
            # token_indices: 哪些 token 选中了它
            # expert_mask[expert_idx] shape: (total_tokens, k)
            idx_in_topk = torch.where(expert_mask[expert_idx] > 0)

            if idx_in_topk[0].numel() == 0:
                continue  # 没有 token 选中这个专家

            # 提取需要该专家处理的 token inputs
            token_indices = idx_in_topk[0]
            current_state = x_flat[token_indices]

            # 专家前向计算
            expert_output = self.experts[expert_idx](current_state)

            # 加权累加到最终输出
            # 我们需要找到对应的权重。
            # routing_weights 形状是 (total_tokens, k)
            # idx_in_topk[1] 给出了在 k 维度上的索引
            weights = routing_weights[token_indices, idx_in_topk[1]].unsqueeze(-1)

            # final_output[token_indices] += expert_output * weights
            # 注意：原地操作 += 在某些梯度场景可能有问题，使用 index_add_ 更安全
            final_output.index_add_(0, token_indices, expert_output * weights)

        return final_output.view(batch_size, seq_len, dim)

# END OF FILE: models/blocks/feedforward/moe.py