# FILE: align/algorithms/orpo/implementation.py
# -*- coding: utf-8 -*-
"""
[新增] ORPO (Odds Ratio Preference Optimization) 核心损失函数实现。
"""
import torch
import torch.nn.functional as F


def orpo_loss(
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        chosen_labels: torch.Tensor,
        alpha: float,
) -> torch.Tensor:
    """
    计算 ORPO 损失。
    ORPO Loss = SFT Loss (on chosen) + Preference Loss
    """
    # 1. SFT Loss (负对数似然损失)
    # 只在 chosen 回答上计算
    sft_losses = -get_log_probs(policy_chosen_logits, chosen_labels)
    sft_loss = sft_losses.mean()

    # 2. Preference Loss (基于 log odds ratio)
    # Log Odds Ratio = log( pi_chosen / ref_chosen ) - log( pi_rejected / ref_rejected )
    #                = (log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected)

    log_odds_chosen = (policy_chosen_logps - reference_chosen_logps).mean(-1)
    log_odds_rejected = (policy_rejected_logps - reference_rejected_logps).mean(-1)

    odds_ratio = log_odds_chosen - log_odds_rejected

    preference_loss = -F.logsigmoid(odds_ratio).mean()

    # 最终损失是两者的加权和，但在原始论文中权重为1
    # 这里的 alpha 是 SFT loss 的系数
    return (alpha * sft_loss) + preference_loss


def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """辅助函数，计算给定logits下标签的对数概率。"""
    # labels 已经是 one-hot 编码的了，或者我们可以用 gather
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

# END OF FILE: align/algorithms/orpo/implementation.py