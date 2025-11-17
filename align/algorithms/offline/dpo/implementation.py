# FILE: align/algorithms/offline/dpo/implementation.py
# -*- coding: utf-8 -*-
"""
[v1.3 - 教学增强版] DPO (Direct Preference Optimization) 核心损失函数实现。
- 增加了注释，将代码与原论文的数学公式对应。
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
    计算 DPO 损失, 理论与代码对应:

    1. **策略与参考模型的对数概率比 (Log Ratios)**:
       - `pi_logratios` = log(pi_policy(chosen) / pi_policy(rejected))
       - `ref_logratios` = log(pi_ref(chosen) / pi_ref(rejected))

       这对应于论文中对隐式奖励 R 的估计。
       R(x, y) ≈ beta * log(pi_policy(y|x) / pi_ref(y|x))

    2. **Logits**:
       - `logits` = `pi_logratios` - `ref_logratios`
       - 这等于 `beta * log( (pi_policy(chosen)/pi_ref(chosen)) / (pi_policy(rejected)/pi_ref(rejected)) )`
       - 这实际上是模型认为 "chosen" 比 "rejected" 好多少的对数几率。

    3. **损失函数**:
       - `loss` = -log_sigmoid(logits)
       - 这是一个标准的 logistic loss，目标是最大化 `logits`，即让模型更倾向于 "chosen" 回答。
    """
    # [核心修改] 在求和之前应用掩码，只计算有效token(回答部分)的logp
    policy_chosen_logps = (policy_chosen_logps * chosen_mask).sum(-1)
    policy_rejected_logps = (policy_rejected_logps * rejected_mask).sum(-1)
    reference_chosen_logps = (reference_chosen_logps * chosen_mask).sum(-1)
    reference_rejected_logps = (reference_rejected_logps * rejected_mask).sum(-1)

    # 计算策略模型和参考模型各自对于 chosen vs rejected 的对数概率差异
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO的核心：计算策略模型相对于参考模型的改进程度
    logits = pi_logratios - ref_logratios

    # 应用 logistic loss
    if label_smoothing > 0:
        loss_chosen = -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        loss_rejected = -F.logsigmoid(-beta * logits) * label_smoothing
        loss = (loss_chosen + loss_rejected).mean()
    else:
        loss = -F.logsigmoid(beta * logits).mean()

    return loss

# END OF FILE: align/algorithms/offline/dpo/implementation.py