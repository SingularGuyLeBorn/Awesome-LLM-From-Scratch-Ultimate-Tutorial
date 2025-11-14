# FILE: pretrain/components/hooks.py
"""
实现用于监控模型内部状态的PyTorch钩子（Hooks）。
"""
import torch
import torch.nn as nn
from typing import List, Dict


class ForwardHook:
    """
    一个前向钩子，用于捕获模块输出激活值的L2范数。
    """

    def __init__(self, name: str):
        self.name = name
        self.norm = None

    def __call__(self, module, input, output):
        # output可能是元组（例如LSTM），我们只关心第一个张量
        if isinstance(output, tuple):
            output = output[0]
        self.norm = torch.norm(output.detach().float()).item()

    def get_metric(self) -> Dict[str, float]:
        if self.norm is not None:
            return {f"activations/{self.name}": self.norm}
        return {}


class BackwardHook:
    """
    一个反向钩子，用于捕获模块权重梯度的L2范数。
    """

    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.norm = None

    def __call__(self, grad):
        # 这个钩子附加在张量上，所以它的输入直接是梯度
        self.norm = torch.norm(grad.detach().float()).item()

    def get_metric(self) -> Dict[str, float]:
        if self.norm is not None:
            return {f"gradients/{self.name}": self.norm}
        return {}


def register_hooks(model: nn.Module) -> List:
    """
    遍历模型，为指定的层（Attention和FeedForward）注册前向和反向钩子。

    Returns:
        List: 包含所有已注册钩子对象的列表，以便后续收集指标。
    """
    hooks = []

    # 遍历所有TransformerBlock
    for i, block in enumerate(model.layers):
        # --- 为Attention层注册钩子 ---
        # 1. 激活值钩子 (前向)
        attn_fwd_hook = ForwardHook(name=f"layer_{i}/attention")
        block.attention.register_forward_hook(attn_fwd_hook)
        hooks.append(attn_fwd_hook)

        # 2. 梯度钩子 (反向) - 附加在输出投影层的权重上
        attn_bwd_hook = BackwardHook(name=f"layer_{i}/attention.wo", module=block.attention.wo)
        block.attention.wo.weight.register_hook(attn_bwd_hook)
        hooks.append(attn_bwd_hook)

        # --- 为FeedForward层注册钩子 ---
        # 1. 激活值钩子 (前向)
        ffn_fwd_hook = ForwardHook(name=f"layer_{i}/feed_forward")
        block.feed_forward.register_forward_hook(ffn_fwd_hook)
        hooks.append(ffn_fwd_hook)

        # 2. 梯度钩子 (反向) - 附加在向下投影层的权重上
        ffn_bwd_hook = BackwardHook(name=f"layer_{i}/feed_forward.w_down", module=block.feed_forward.w_down)
        block.feed_forward.w_down.weight.register_hook(ffn_bwd_hook)
        hooks.append(ffn_bwd_hook)

    return hooks

# END OF FILE: pretrain/components/hooks.py