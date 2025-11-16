# FILE: align/algorithms/online/gspo/implementation.py
"""
[v2.0 - 理论完备版] GSPO (Group Sequence Policy Optimization) 核心损失函数实现。
- 修正为 PPO 的 Clipped Surrogate Objective 形式。
"""
import torch


def gspo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float
) -> torch.Tensor:
    """
    计算 GSPO 损失 (与PPO损失函数结构一致，但作用于序列级)。
    Args:
        log_probs: 当前策略在序列上的对数概率 (batch_size,)
        old_log_probs: 旧策略在序列上的对数概率 (batch_size,)
        advantages: 序列级别的优势 (batch_size,)
        clip_epsilon: 裁剪范围
    Returns:
        GSPO 策略损失
    """
    # 计算序列级别的重要性采样比率 s_i(theta)
    ratios = torch.exp(log_probs - old_log_probs)

    # 计算 unclipped objective
    surr1 = ratios * advantages

    # 计算 clipped objective
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # [核心修复] 应用 PPO 的 Clipped Surrogate Objective，取两者中的较小者
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss