# FILE: align/algorithms/online/ppo/implementation.py
# -*- coding: utf-8 -*-
"""
[v2.3 - 数值稳定版] PPO (Proximal Policy Optimization) 核心算法实现。
- 修复: 在 Batch 内有效 Token 极少时，避免除以极小数值导致的梯度爆炸。
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
    使用 GAE (Generalized Advantage Estimation) 计算优势和回报。

    核心公式 (通过向后迭代高效实现):
    - TD-error (delta_t) = r_t + gamma * V(s_{t+1}) - V(s_t)
    - Advantage (A_t)   = delta_t + gamma * lambda * A_{t+1}

    Args:
        rewards: (batch_size, seq_len)
        values: (batch_size, seq_len)
        mask: (batch_size, seq_len)
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    # 从序列的最后一个时间步向前迭代
    for t in reversed(range(rewards.size(1))):
        # 如果是最后一个step，下一个value是0；否则是下一个时间步的value
        next_value = values[:, t + 1] if t < rewards.size(1) - 1 else 0.0

        # 计算 TD-error (delta)
        # mask[:, t] 用于处理 padding，确保 padding 处的 value 不参与计算
        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # 计算 GAE 优势
        # mask[:, t] 确保我们只在非填充部分累积优势，并且阻断跨序列的错误传播
        last_advantage = delta + gamma * lambda_gae * last_advantage * mask[:, t]
        advantages[:, t] = last_advantage

    # 计算回报 (Returns), 即价值函数的目标
    returns = advantages + values

    return advantages, returns


def ppo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        clip_epsilon: float
) -> torch.Tensor:
    """
    计算 PPO 的裁剪代理目标损失 (Clipped Surrogate Objective)。

    [关键修复]: 增加了对 mask.sum() 的检查，防止除零或数值不稳定。
    """
    # 计算重要性采样比率 pi_theta / pi_old
    ratios = torch.exp(log_probs - old_log_probs)

    # PPO 核心逻辑: Min(surr1, surr2)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # 计算未裁剪的、每个token的损失 (最大化目标 -> 最小化负值)
    # (bs, seq_len)
    loss_unmasked = -torch.min(surr1, surr2)

    # 应用掩码，只计算有效token的损失
    loss_masked = loss_unmasked * mask

    # [核心修复] 安全求平均
    # 在 batch_size 较小或 padding 很多的情况下，mask_sum 可能接近 0
    mask_sum = mask.sum()

    # 如果有效 token 数太少，直接返回 0 损失，避免梯度爆炸
    # 这里使用 1.0 作为阈值，意味着至少要有 1 个有效 token
    if mask_sum < 1.0:
        # 返回一个带有 grad_fn 的 0 张量，确保计算图完整性
        return loss_masked.sum() * 0.0

    return loss_masked.sum() / mask_sum

# END OF FILE: align/algorithms/online/ppo/implementation.py