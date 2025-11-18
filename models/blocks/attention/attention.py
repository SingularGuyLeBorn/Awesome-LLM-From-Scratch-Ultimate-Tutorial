# FILE: models/blocks/attention/attention.py
"""
【Attention Museum - 入口文件】
这是一个工厂包装器，负责根据配置 (args.attention_variant)
实例化并调用具体的注意力实现。

具体实现位于同级目录的:
- standard.py (MHA, GQA, MQA, MLA)
- linear.py (Linear Attention)
- sparse.py (MoBA)
"""
import torch.nn as nn
from models.config import ModelArgs

# 导入各个流派的实现
from .standard import StandardAttention, MultiHeadLatentAttention
from .linear import LinearAttention
from .sparse import MixtureOfBlockAttention


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
        else:
            raise ValueError(f"Unknown attention variant: {self.variant}")

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        """
        统一的前向传播接口。
        显式接收 paged_attention_inputs 以避免参数冲突。
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