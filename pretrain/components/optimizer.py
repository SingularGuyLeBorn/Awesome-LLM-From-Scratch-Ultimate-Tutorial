# FILE: pretrain/components/optimizer.py
"""
【v1.1 - 健壮性增强】
- 增加检查，防止在没有可训练参数时崩溃。
"""
import torch
import torch.nn as nn
import logging


def get_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    """
    为模型创建AdamW优化器，并对参数进行分组以应用不同的权重衰减。
    """
    # 只收集需要梯度的参数
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # [核心修复] 检查是否存在可训练参数
    if not param_dict:
        raise ValueError("优化器错误：模型中没有找到任何可训练的参数 (requires_grad=True)。"
                         "如果您正在使用LoRA，请检查LoRA层是否已成功应用。")

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    total_trainable = num_decay_params + num_nodecay_params

    logging.info(f"优化器参数分组 (总可训练参数: {total_trainable:,}):")
    logging.info(
        f"  - 带权重衰减 (decay): {num_decay_params:,} ({num_decay_params / total_trainable:.2%})")
    logging.info(f"  - 不带权重衰减 (no decay): {num_nodecay_params:,}")

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    return optimizer

# END OF FILE: pretrain/components/optimizer.py