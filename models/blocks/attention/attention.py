# FILE: models/blocks/attention/attention.py
"""
【Attention Museum - v3.4 NSA 集成版】
- 在工厂类中新增 "nsa" 选项。
"""
import torch.nn as nn
from models.config import ModelArgs

# 导入各个流派的实现
from .standard import StandardAttention, MultiHeadLatentAttention
from .linear import LinearAttention
from .sparse import MixtureOfBlockAttention, NativeSparseAttention


class Attention(nn.Module):
    """
    通用 Attention 包装器。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.variant = args.attention_variant

        if self.variant == "mha":
            self.impl = StandardAttention(args)
        elif self.variant == "mla":
            self.impl = MultiHeadLatentAttention(args)
        elif self.variant == "linear":
            self.impl = LinearAttention(args)
        elif self.variant == "moba":
            self.impl = MixtureOfBlockAttention(args)
        elif self.variant == "nsa":
            self.impl = NativeSparseAttention(args)
        else:
            raise ValueError(f"Unknown attention variant: {self.variant}")

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        """
        统一的前向传播接口。
        """
        return self.impl(
            x,
            rope,
            layer_idx,
            kv_cache=kv_cache,
            start_pos=start_pos,
            paged_attention_inputs=paged_attention_inputs,
            **kwargs
        )

# END OF FILE: models/blocks/attention/attention.py