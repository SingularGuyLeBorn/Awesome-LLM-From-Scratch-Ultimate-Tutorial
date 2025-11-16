# FILE: align/algorithms/gspo/implementation.py
"""
[新增] GSPO (Group Sequence Policy Optimization) 核心损失函数实现。
严格遵循 Qwen3 技术报告中的定义。
"""
import torch


def gspo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float
) -> torch.Tensor:
    """
    计算 GSPO 损失。
    Args:
        log_probs: 当前策略在序列上的对数概率 (batch_size,)
        old_log_probs: 旧策略在序列上的对数概率 (batch_size,)
        advantages: 标准化的奖励 (batch_size,)
        clip_epsilon: 裁剪范围
    Returns:
        GSPO 策略损失
    """
    # 计算序列级别的重要性采样比率 s_i(theta)
    sequence_ratios = torch.exp(log_probs - old_log_probs)

    # 根据优势 A_i 的符号决定裁剪边界
    # torch.where(condition, x, y) is like "if condition x else y"
    clipped_ratios = torch.where(
        advantages > 0,
        torch.clamp(sequence_ratios, min=1.0 - clip_epsilon),
        torch.clamp(sequence_ratios, max=1.0 + clip_epsilon)
    )

    # 根据论文公式 (5) 的简化版 (不带clip)，损失是 -s_i(theta) * A_i
    # 这里我们使用裁剪后的版本
    loss = - (clipped_ratios * advantages).mean()

    return loss
# END OF FILE: align/algorithms/gspo/implementation.py