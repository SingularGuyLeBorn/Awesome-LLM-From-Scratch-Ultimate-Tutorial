# FILE: models/reward_model.py
"""
[v1.2 - RM训练专用版] 奖励模型 (Reward Model) 的架构。
- forward 方法现在只返回最终的标量奖励，以简化RM训练和PPO流程。
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播，只返回最终的序列标量奖励。
        Args:
            tokens (torch.Tensor): 输入的 token ID 序列, 形状 (batch_size, seq_len)
        Returns:
            end_rewards (torch.Tensor): 每个序列的最终标量奖励, 形状 (batch_size,)
        """
        hidden_states = self.transformer(tokens, return_hidden_states=True)

        # (batch_size, seq_len, 1)
        all_rewards = self.reward_head(hidden_states)

        # 找到每个序列中最后一个非填充 token 的位置
        # 假设 padding_id 是 0, 或者任何在序列末尾不会自然出现的token
        # 一个更健壮的方法是接收 padding_mask
        non_pad_mask = tokens != 0
        last_token_indices = non_pad_mask.long().cumsum(dim=1).argmax(dim=1)

        batch_indices = torch.arange(all_rewards.size(0), device=all_rewards.device)

        # (batch_size,)
        end_rewards = all_rewards[batch_indices, last_token_indices].squeeze(-1)

        return end_rewards
# END OF FILE: models/reward_model.py