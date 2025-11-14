# FILE: pretrain/components/optimizer.py
"""
创建AdamW优化器，并应用权重衰减（weight decay）分离。
"""
import torch
import torch.nn as nn
import logging


def get_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    """
    为模型创建AdamW优化器，并对参数进行分组以应用不同的权重衰减。
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # 将所有2D以上的参数（通常是权重矩阵）归入需要权重衰减的组
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # 其余参数（如LayerNorm的权重、偏置等）不进行权重衰减
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logging.info(f"优化器参数分组:")
    logging.info(
        f"  - 带权重衰减 (decay) 的参数: {num_decay_params:,} (占总参数的 {num_decay_params / (num_decay_params + num_nodecay_params):.2%})")
    logging.info(f"  - 不带权重衰减 (no decay) 的参数: {num_nodecay_params:,}")

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    return optimizer

# END OF FILE: pretrain/components/optimizer.py