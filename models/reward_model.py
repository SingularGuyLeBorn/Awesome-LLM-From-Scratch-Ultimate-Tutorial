# FILE: models/reward_model.py
"""
[v1.3 - 健壮性修复版] 奖励模型 (Reward Model) 的架构。
- forward 方法现在接收 attention_mask 以精确地定位最后一个 token。
"""
import torch
import torch.nn as nn
from .transformer import Transformer
from .config import ModelArgs


class RewardModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.transformer = Transformer(args)
        self.reward_head = nn.Linear(args.dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，只返回最终的序列标量奖励。

        Args:
            tokens (torch.Tensor): 输入的 token ID 序列, 形状 (batch_size, seq_len)
            attention_mask (torch.Tensor): 注意力掩码, 形状 (batch_size, seq_len), 1表示真实token, 0表示padding

        Returns:
            end_rewards (torch.Tensor): 每个序列的最终标量奖励, 形状 (batch_size,)
        """
        hidden_states = self.transformer(tokens, return_hidden_states=True)

        # (batch_size, seq_len, 1)
        all_rewards = self.reward_head(hidden_states)

        # [核心修复] 使用 attention_mask 来精确定位最后一个真实 token 的位置
        # 1. 计算每个序列的真实长度
        sequence_lengths = attention_mask.sum(dim=1, dtype=torch.long)

        # 2. 最后一个 token 的索引是长度减一
        last_token_indices = sequence_lengths - 1

        # 3. 创建批次索引
        batch_indices = torch.arange(all_rewards.size(0), device=all_rewards.device)

        # 4. 使用高级索引从 all_rewards 中提取每个序列的最后一个真实 token 的奖励值
        # (batch_size,)
        end_rewards = all_rewards[batch_indices, last_token_indices].squeeze(-1)

        return end_rewards
# END OF FILE: models/reward_model.py