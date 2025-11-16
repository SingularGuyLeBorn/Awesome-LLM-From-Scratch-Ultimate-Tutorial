# FILE: align/algorithms/ppo/implementation.py
# -*- coding: utf-8 -*-
"""
[v2.0 - 理论完备版] PPO (Proximal Policy Optimization) 核心算法实现。
- 实现了完整版的 GAE (Generalized Advantage Estimation)。
"""
import torch
from typing import Tuple


def compute_advantages(
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gamma: float = 0.99,
        lambda_gae: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [核心重构] 使用 GAE (Generalized Advantage Estimation) 计算优势和回报。
    严格遵循原始论文和主流库 (TRL) 的实现。

    Args:
        rewards: 奖励张量, 形状 (batch_size, seq_len)
        values: 价值模型输出的价值张量, 形状 (batch_size, seq_len)
        mask: 掩码张量，标记非填充部分, 形状 (batch_size, seq_len)
        gamma: 折扣因子
        lambda_gae: GAE lambda 参数

    Returns:
        advantages (torch.Tensor): 优势张量, 形状 (batch_size, seq_len)
        returns (torch.Tensor): 回报张量 (即价值目标), 形状 (batch_size, seq_len)
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    # 从序列的最后一个时间步向前迭代
    for t in reversed(range(rewards.size(1))):
        # 如果是最后一个step，下一个value是0；否则是下一个时间步的value
        next_value = values[:, t + 1] if t < rewards.size(1) - 1 else 0.0

        # 计算 TD-error (delta)
        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # 计算 GAE 优势
        # A_t = delta_t + gamma * lambda * A_{t+1}
        # mask[:, t] 确保我们只在非填充部分累积优势
        last_advantage = delta + gamma * lambda_gae * last_advantage * mask[:, t]
        advantages[:, t] = last_advantage

    # 计算回报 (Returns), 即价值函数的目标
    # returns = advantages + values
    returns = advantages + values

    return advantages, returns


def ppo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float
) -> torch.Tensor:
    """
    计算 PPO 的裁剪代理目标损失 (与之前版本保持一致，理论正确)。
    """
    # 计算重要性采样比率
    ratios = torch.exp(log_probs - old_log_probs)

    # PPO 核心逻辑
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # 取两者中的较小者
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss
# END OF FILE: align/algorithms/ppo/implementation.py