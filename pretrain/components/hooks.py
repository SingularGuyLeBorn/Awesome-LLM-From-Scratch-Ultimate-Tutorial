# FILE: pretrain/components/hooks.py
"""
实现用于监控模型内部状态的PyTorch钩子（Hooks）。
【v2.0 - MoE 适配版】
- 修复了 MoE 层没有 `w_down` 属性导致的崩溃问题。
- 针对 MoE 层，改为监控 Router 的梯度，以观察专家选择的学习情况。
"""
import torch
import torch.nn as nn
from typing import List, Dict

# 动态导入 MoELayer 以进行类型检查，避免循环导入
try:
    from models.blocks.feedforward.moe import MoELayer
except ImportError:
    MoELayer = None


class ForwardHook:
    """
    一个前向钩子，用于捕获模块输出激活值的详细统计数据。
    """

    def __init__(self, name: str):
        self.name = name
        self.stats = {}  # 使用字典存储所有统计数据

    def __call__(self, module, input, output):
        # output可能是元组（例如LSTM），我们只关心第一个张量
        if isinstance(output, tuple):
            output = output[0]

        tensor = output.detach().float()
        # 计算更详细的统计信息
        self.stats = {
            f"activations/{self.name}_norm": torch.norm(tensor).item(),
            f"activations/{self.name}_max": tensor.max().item(),
            f"activations/{self.name}_min": tensor.min().item(),
            f"activations/{self.name}_mean": tensor.mean().item()
        }

    def get_metric(self) -> Dict[str, float]:
        return self.stats  # 返回完整的统计字典


class BackwardHook:
    """
    一个反向钩子，用于捕获模块权重梯度的详细统计数据。
    """

    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.stats = {}

    def __call__(self, grad):
        # 这个钩子附加在张量上，所以它的输入直接是梯度
        if grad is None:
            return

        tensor = grad.detach().float()
        self.stats = {
            f"gradients/{self.name}_norm": torch.norm(tensor).item(),
            f"gradients/{self.name}_max": tensor.max().item(),
            f"gradients/{self.name}_min": tensor.min().item(),
            f"gradients/{self.name}_mean": tensor.mean().item()
        }

    def get_metric(self) -> Dict[str, float]:
        return self.stats


def register_hooks(model: nn.Module) -> List:
    """
    遍历模型，为所有关键层注册前向和反向钩子，以实现全面的“内科”监控。

    Returns:
        List: 包含所有已注册钩子对象的列表，以便后续收集指标。
    """
    hooks = []

    # 1. 监控 Embedding 层的梯度
    if hasattr(model, 'tok_embeddings') and hasattr(model.tok_embeddings, 'weight'):
        emb_bwd_hook = BackwardHook(name="tok_embeddings", module=model.tok_embeddings)
        if model.tok_embeddings.weight.requires_grad:
            model.tok_embeddings.weight.register_hook(emb_bwd_hook)
            hooks.append(emb_bwd_hook)

    # 2. 遍历所有TransformerBlock
    for i, block in enumerate(model.layers):
        # --- 为Attention层注册钩子 ---
        # 激活值钩子 (前向)
        attn_fwd_hook = ForwardHook(name=f"layer_{i}/attention.output")
        block.attention.register_forward_hook(attn_fwd_hook)
        hooks.append(attn_fwd_hook)

        # 梯度钩子 (反向)
        attn_bwd_hook = BackwardHook(name=f"layer_{i}/attention.wo", module=block.attention.wo)
        block.attention.wo.weight.register_hook(attn_bwd_hook)
        hooks.append(attn_bwd_hook)

        # --- 为FeedForward层注册钩子 (MoE 适配逻辑) ---
        # 激活值钩子 (前向) - 无论是 Dense 还是 MoE，输出监控逻辑一致
        ffn_fwd_hook = ForwardHook(name=f"layer_{i}/feed_forward.output")
        block.feed_forward.register_forward_hook(ffn_fwd_hook)
        hooks.append(ffn_fwd_hook)

        # 梯度钩子 (反向) - 需要区分 MoE 和 Dense
        if MoELayer and isinstance(block.feed_forward, MoELayer):
            # 对于 MoE，我们监控 Router (门控网络) 的权重梯度
            # 这能反映模型是否在学习如何分配专家
            router_bwd_hook = BackwardHook(name=f"layer_{i}/moe_router", module=block.feed_forward.router)
            if block.feed_forward.router.weight.requires_grad:
                block.feed_forward.router.weight.register_hook(router_bwd_hook)
                hooks.append(router_bwd_hook)
        else:
            # 对于 Dense FFN，监控 w_down
            if hasattr(block.feed_forward, 'w_down'):
                ffn_bwd_hook = BackwardHook(name=f"layer_{i}/feed_forward.w_down", module=block.feed_forward.w_down)
                if block.feed_forward.w_down.weight.requires_grad:
                    block.feed_forward.w_down.weight.register_hook(ffn_bwd_hook)
                    hooks.append(ffn_bwd_hook)

        # --- 监控 Normalization 层的激活值和梯度 ---
        # 3a. Attention Norm
        attn_norm_fwd_hook = ForwardHook(name=f"layer_{i}/attention_norm")
        block.attention_norm.register_forward_hook(attn_norm_fwd_hook)
        hooks.append(attn_norm_fwd_hook)

        attn_norm_bwd_hook = BackwardHook(name=f"layer_{i}/attention_norm.weight", module=block.attention_norm)
        if block.attention_norm.weight.requires_grad:
            block.attention_norm.weight.register_hook(attn_norm_bwd_hook)
            hooks.append(attn_norm_bwd_hook)

        # 3b. FFN Norm
        ffn_norm_fwd_hook = ForwardHook(name=f"layer_{i}/ffn_norm")
        block.ffn_norm.register_forward_hook(ffn_norm_fwd_hook)
        hooks.append(ffn_norm_fwd_hook)

        ffn_norm_bwd_hook = BackwardHook(name=f"layer_{i}/ffn_norm.weight", module=block.ffn_norm)
        if block.ffn_norm.weight.requires_grad:
            block.ffn_norm.weight.register_hook(ffn_norm_bwd_hook)
            hooks.append(ffn_norm_bwd_hook)

    # 3. 监控最终的 Normalization 层
    if hasattr(model, 'norm'):
        final_norm_fwd_hook = ForwardHook(name="final_norm")
        model.norm.register_forward_hook(final_norm_fwd_hook)
        hooks.append(final_norm_fwd_hook)

        final_norm_bwd_hook = BackwardHook(name="final_norm.weight", module=model.norm)
        if model.norm.weight.requires_grad:
            model.norm.weight.register_hook(final_norm_bwd_hook)
            hooks.append(final_norm_bwd_hook)

    # 4. 监控最终 LM Head 的梯度
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_bwd_hook = BackwardHook(name="lm_head", module=model.lm_head)
        if model.lm_head.weight.requires_grad:
            model.lm_head.weight.register_hook(lm_head_bwd_hook)
            hooks.append(lm_head_bwd_hook)

    return hooks
# END OF FILE: pretrain/components/hooks.py