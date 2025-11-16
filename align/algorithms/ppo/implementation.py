# FILE: align/algorithms/ppo/implementation.py
"""
[v1.1 - 重构版] PPO (Proximal Policy Optimization) 核心算法实现。
- 优势计算 (GAE) 逻辑被重构，以处理RLHF中的标量奖励。
"""
import torch
from typing import Tuple

def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> torch.Tensor:
    """
    [核心重构] 使用 GAE (Generalized Advantage Estimation) 计算优势。
    此版本适用于 RLHF 场景，其中每个序列只有一个最终的标量奖励。
    Args:
        rewards: 标量奖励张量, 形状 (num_sequences,)
        values: 序列最后一个时间步的价值估计, 形状 (num_sequences,)
        gamma: 折扣因子
        lambda_gae: GAE lambda 参数
    Returns:
        advantages: 标量优势张量, 形状 (num_sequences,)
    """
    # 在RLHF中，我们通常将奖励视为在序列结束时一次性获得的
    # 并且价值函数 V(s_last) 是对未来奖励的期望
    # GAE在这种情况下可以简化
    advantages = rewards - values
    # 注意：更复杂的实现可能会考虑KL惩罚项等，这里我们保持简单
    return advantages

def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float
) -> torch.Tensor:
    """
    计算 PPO 的裁剪代理目标损失。
    """
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss
# END OF FILE: align/algorithms/ppo/implementation.py