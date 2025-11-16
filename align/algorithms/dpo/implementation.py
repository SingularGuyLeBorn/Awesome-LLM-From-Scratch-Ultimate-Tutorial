# FILE: align/algorithms/dpo/implementation.py
"""
[新增] DPO (Direct Preference Optimization) 核心损失函数实现。
"""
import torch
import torch.nn.functional as F


def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float
) -> torch.Tensor:
    """
    计算 DPO 损失。
    Args:
        policy_chosen_logps: policy模型对chosen回答的对数概率
        policy_rejected_logps: policy模型对rejected回答的对数概率
        reference_chosen_logps: reference模型对chosen回答的对数概率
        reference_rejected_logps: reference模型对rejected回答的对数概率
        beta: DPO的温度超参数
    Returns:
        损失张量
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    return loss
# END OF FILE: align/algorithms/dpo/implementation.py