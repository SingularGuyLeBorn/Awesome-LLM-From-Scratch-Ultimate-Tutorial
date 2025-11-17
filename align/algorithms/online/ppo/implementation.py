# FILE: align/algorithms/online/ppo/implementation.py
# -*- coding: utf-8 -*-
"""
[v2.2 - 掩码修复版] PPO (Proximal Policy Optimization) 核心算法实现。
- 核心修复: ppo_loss 现在接收一个掩码(mask)，确保损失只在有效token上计算。
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

    这个函数正是实现了上述递归公式。
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
        # mask[:, t] 确保我们只在非填充部分累积优势
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
    [核心修改] 新增了 mask 参数。
    """
    # 计算重要性采样比率
    ratios = torch.exp(log_probs - old_log_probs)

    # PPO 核心逻辑
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # 计算未裁剪的、每个token的损失
    # (bs, seq_len)
    loss_unmasked = -torch.min(surr1, surr2)

    # 应用掩码，只计算有效token的损失
    loss_masked = loss_unmasked * mask

    # 对有效token的损失取平均
    # 加上一个极小值防止 mask.sum() 为0
    return loss_masked.sum() / (mask.sum() + 1e-8)
# END OF FILE: align/algorithms/online/ppo/implementation.py