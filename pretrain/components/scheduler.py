# FILE: pretrain/components/scheduler.py
"""
创建学习率调度器。
"""
import torch
import math


def get_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_iters: int, max_iters: int, min_lr: float):
    """
    创建带预热的余弦退火学习率调度器。

    Args:
        optimizer: 优化器。
        warmup_iters: 预热阶段的迭代次数。
        max_iters: 总的训练迭代次数。
        min_lr: 最小学习率。

    Returns:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器。
    """

    # 学习率函数
    def lr_lambda(current_iter: int):
        # 1) 线性预热
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        # 2) 如果超过最大迭代次数，则保持最小学习率
        if current_iter > max_iters:
            return min_lr / optimizer.defaults['lr']  # 返回一个比例
        # 3) 余弦退火
        # 计算退火阶段的进度
        decay_ratio = float(current_iter - warmup_iters) / float(max(1, max_iters - warmup_iters))
        assert 0.0 <= decay_ratio <= 1.0
        # 计算余弦值
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        # 计算当前学习率与初始学习率的比例
        return (min_lr + coeff * (optimizer.defaults['lr'] - min_lr)) / optimizer.defaults['lr']

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# END OF FILE: pretrain/components/scheduler.py