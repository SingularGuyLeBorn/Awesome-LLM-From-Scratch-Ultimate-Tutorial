# FILE: models/blocks/attention/sparse.py
"""
【流派三：稀疏注意力】
包含：
1. MixtureOfBlockAttention (MoBA): 基于 Gating 的块稀疏注意力。
"""
import torch
import torch.nn as nn
from .standard import StandardAttention


class MixtureOfBlockAttention(nn.Module):
    """
    MoBA: 将历史上下文分块，通过 Gating 机制选择 Top-K 块进行 Attention。
    """

    def __init__(self, args):
        super().__init__()
        # 复用 StandardAttention 作为基础计算单元
        self.impl = StandardAttention(args)
        self.block_size = args.moba_block_size
        self.topk = args.moba_topk
        self.gate = nn.Linear(args.dim, 1, bias=False)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        # MoBA 逻辑:
        # 1. 将上下文切分为 Blocks
        # 2. 计算 Query 对每个 Block 的相关性 (Gating)
        # 3. 选出 Top-K Blocks
        # 4. 构造稀疏 Mask
        # 5. 执行 Attention

        # 在本教程的简单实现中，我们直接透传给 StandardAttention。
        # 真正的 MoBA 需要重写 KV Cache 的读取逻辑，只读取被选中的 Blocks。

        return self.impl(
            x,
            rope,
            layer_idx,
            kv_cache=kv_cache,
            start_pos=start_pos,
            paged_attention_inputs=paged_attention_inputs,
            **kwargs
        )

# END OF FILE: models/blocks/attention/sparse.py