# FILE: align/algorithms/online/grpo/implementation.py
"""
[新增] GRPO (Group Relative Policy Optimization) 核心损失函数实现。

理论说明:
GRPO 的核心创新在于其“无价值模型”的优势函数计算方法，该方法在训练器 (trainer.py)
的 Rollout 阶段实现。

在模型更新阶段，GRPO 遵循与 PPO 相同的 token-level Clipped Surrogate Objective。
因此，此处的 grpo_loss 函数在结构上与 ppo_loss 完全相同，但我们将其独立出来，
以保持代码结构与算法名称的一一对应，增强可读性和清晰度。
"""
import torch

def grpo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float
) -> torch.Tensor:
    """
    计算 GRPO 的裁剪代理目标损失 (结构上与 PPO 损失一致)。
    Args:
        log_probs: 当前策略在每个token上的对数概率 (batch_size, seq_len)
        old_log_probs: 旧策略在每个token上的对数概率 (batch_size, seq_len)
        advantages: 优势张量 (batch_size, seq_len)，对于GRPO，这是一个序列内恒定的值
        clip_epsilon: 裁剪范围
    Returns:
        GRPO 策略损失
    """
    # 计算重要性采样比率
    ratios = torch.exp(log_probs - old_log_probs)

    # PPO 核心逻辑
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # 取两者中的较小者
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss
# END OF FILE: align/algorithms/online/grpo/implementation.py