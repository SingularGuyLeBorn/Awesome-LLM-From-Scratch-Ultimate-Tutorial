# FILE: align/algorithms/offline/dpo/implementation.py
# -*- coding: utf-8 -*-
"""
[v1.2 - 理论完备版] DPO (Direct Preference Optimization) 核心损失函数实现。
- 引入掩码处理以精确计算序列对数概率。
- 引入标签平滑以提高鲁棒性。
"""
import torch
import torch.nn.functional as F


def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    计算带标签平滑的DPO损失，并只在回答部分进行计算。
    Args:
        ... (logps张量形状为 (batch_size, seq_len))
        beta: DPO的温度超参数
        chosen_mask: chosen回答的掩码, 形状 (batch_size, seq_len)
        rejected_mask: rejected回答的掩码, 形状 (batch_size, seq_len)
        label_smoothing: 标签平滑系数
    Returns:
        损失张量
    """
    # [核心修改] 在求和之前应用掩码，只计算有效token(回答部分)的logp
    policy_chosen_logps = (policy_chosen_logps * chosen_mask).sum(-1)
    policy_rejected_logps = (policy_rejected_logps * rejected_mask).sum(-1)
    reference_chosen_logps = (reference_chosen_logps * chosen_mask).sum(-1)
    reference_rejected_logps = (reference_rejected_logps * rejected_mask).sum(-1)

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # [核心新增] 应用标签平滑
    if label_smoothing > 0:
        loss_chosen = -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        loss_rejected = -F.logsigmoid(-beta * logits) * label_smoothing
        loss = (loss_chosen + loss_rejected).mean()
    else:
        loss = -F.logsigmoid(beta * logits).mean()

    return loss


# END OF FILE: align/algorithms/offline/dpo/implementation.py