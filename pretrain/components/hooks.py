# FILE: pretrain/components/hooks.py
"""
实现用于监控模型内部状态的PyTorch钩子（Hooks）。
【v2.1 - 鲁棒性修复版】
- 修复了 'Attention' object has no attribute 'wo' 错误。
- 增加了智能解包逻辑，能够穿透 Attention Wrapper 和 MoBA Layer 找到底层的 Linear 层。
- 兼容 Dense FFN 和 MoE Router 的监控。
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional

# 动态导入 MoELayer 以进行类型检查
try:
    from models.blocks.feedforward.moe import MoELayer
except ImportError:
    MoELayer = None


class ForwardHook:
    def __init__(self, name: str):
        self.name = name
        self.stats = {}

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        tensor = output.detach().float()
        self.stats = {
            f"activations/{self.name}_norm": torch.norm(tensor).item(),
            f"activations/{self.name}_max": tensor.max().item(),
            f"activations/{self.name}_min": tensor.min().item(),
            f"activations/{self.name}_mean": tensor.mean().item()
        }

    def get_metric(self) -> Dict[str, float]:
        return self.stats


class BackwardHook:
    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.stats = {}

    def __call__(self, grad):
        if grad is None: return
        tensor = grad.detach().float()
        self.stats = {
            f"gradients/{self.name}_norm": torch.norm(tensor).item(),
            f"gradients/{self.name}_max": tensor.max().item(),
            f"gradients/{self.name}_min": tensor.min().item(),
            f"gradients/{self.name}_mean": tensor.mean().item()
        }

    def get_metric(self) -> Dict[str, float]:
        return self.stats


def _unwrap_attention(module: nn.Module) -> nn.Module:
    """
    递归解包 Attention 模块，直到找到具体的实现层 (Standard/MLA/Linear)。
    处理: Attention(Wrapper) -> [MoBA] -> Implementation
    """
    # 如果有 impl 属性，说明是 Wrapper (Attention class) 或 MoBA
    if hasattr(module, 'impl'):
        return _unwrap_attention(module.impl)
    return module


def register_hooks(model: nn.Module) -> List:
    hooks = []

    # 1. Embedding Gradient
    if hasattr(model, 'tok_embeddings') and hasattr(model.tok_embeddings, 'weight'):
        emb_bwd_hook = BackwardHook(name="tok_embeddings", module=model.tok_embeddings)
        if model.tok_embeddings.weight.requires_grad:
            model.tok_embeddings.weight.register_hook(emb_bwd_hook)
            hooks.append(emb_bwd_hook)

    # 2. Layers
    for i, block in enumerate(model.layers):
        # --- Attention Hooks ---
        # 1. Unpack to get the real attention implementation
        real_attn = _unwrap_attention(block.attention)

        # 2. Forward Hook (Output)
        # 我们 hook 在 wrapper 上，这样能捕获最终输出
        attn_fwd_hook = ForwardHook(name=f"layer_{i}/attention.output")
        block.attention.register_forward_hook(attn_fwd_hook)
        hooks.append(attn_fwd_hook)

        # 3. Backward Hook (Projection Weights)
        # 尝试寻找 wo (Standard, MLA, Linear 都有 wo)
        if hasattr(real_attn, 'wo'):
            attn_bwd_hook = BackwardHook(name=f"layer_{i}/attention.wo", module=real_attn.wo)
            if real_attn.wo.weight.requires_grad:
                real_attn.wo.weight.register_hook(attn_bwd_hook)
                hooks.append(attn_bwd_hook)

        # --- FFN Hooks ---
        ffn_fwd_hook = ForwardHook(name=f"layer_{i}/feed_forward.output")
        block.feed_forward.register_forward_hook(ffn_fwd_hook)
        hooks.append(ffn_fwd_hook)

        # MoE Router or Dense w_down
        if MoELayer and isinstance(block.feed_forward, MoELayer):
            router_bwd_hook = BackwardHook(name=f"layer_{i}/moe_router", module=block.feed_forward.router)
            if block.feed_forward.router.weight.requires_grad:
                block.feed_forward.router.weight.register_hook(router_bwd_hook)
                hooks.append(router_bwd_hook)
        else:
            if hasattr(block.feed_forward, 'w_down'):
                ffn_bwd_hook = BackwardHook(name=f"layer_{i}/feed_forward.w_down", module=block.feed_forward.w_down)
                if block.feed_forward.w_down.weight.requires_grad:
                    block.feed_forward.w_down.weight.register_hook(ffn_bwd_hook)
                    hooks.append(ffn_bwd_hook)

        # --- Norm Hooks ---
        # Attention Norm
        attn_norm_fwd_hook = ForwardHook(name=f"layer_{i}/attention_norm")
        block.attention_norm.register_forward_hook(attn_norm_fwd_hook)
        hooks.append(attn_norm_fwd_hook)

        attn_norm_bwd_hook = BackwardHook(name=f"layer_{i}/attention_norm.weight", module=block.attention_norm)
        if hasattr(block.attention_norm, 'weight') and block.attention_norm.weight.requires_grad:
            block.attention_norm.weight.register_hook(attn_norm_bwd_hook)
            hooks.append(attn_norm_bwd_hook)

        # FFN Norm
        ffn_norm_fwd_hook = ForwardHook(name=f"layer_{i}/ffn_norm")
        block.ffn_norm.register_forward_hook(ffn_norm_fwd_hook)
        hooks.append(ffn_norm_fwd_hook)

        ffn_norm_bwd_hook = BackwardHook(name=f"layer_{i}/ffn_norm.weight", module=block.ffn_norm)
        if hasattr(block.ffn_norm, 'weight') and block.ffn_norm.weight.requires_grad:
            block.ffn_norm.weight.register_hook(ffn_norm_bwd_hook)
            hooks.append(ffn_norm_bwd_hook)

    # 3. Final Norm
    if hasattr(model, 'norm'):
        final_norm_fwd_hook = ForwardHook(name="final_norm")
        model.norm.register_forward_hook(final_norm_fwd_hook)
        hooks.append(final_norm_fwd_hook)

        final_norm_bwd_hook = BackwardHook(name="final_norm.weight", module=model.norm)
        if hasattr(model.norm, 'weight') and model.norm.weight.requires_grad:
            model.norm.weight.register_hook(final_norm_bwd_hook)
            hooks.append(final_norm_bwd_hook)

    # 4. LM Head
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_bwd_hook = BackwardHook(name="lm_head", module=model.lm_head)
        if model.lm_head.weight.requires_grad:
            model.lm_head.weight.register_hook(lm_head_bwd_hook)
            hooks.append(lm_head_bwd_hook)

    return hooks
# END OF FILE: pretrain/components/hooks.py