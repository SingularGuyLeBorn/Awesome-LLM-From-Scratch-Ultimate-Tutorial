# FILE: finetune/peft/qlora/linear4bit.py
# -*- coding: utf-8 -*-
"""
[QLoRA Core v2.0] Linear4bit 层实现 (CPU Optimized).
- 修复解量化逻辑：使用查表法 (Lookup) 替代极其缓慢的位运算解包循环。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nf4 import quantize_blockwise, get_nf4_codebook


class Linear4bit(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, block_size: int = 64,
                 compute_dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.compute_dtype = compute_dtype

        # 存储打包后的权重 (uint8)
        self.register_buffer('weight_packed', torch.empty(0, dtype=torch.uint8))
        self.register_buffer('weight_absmax', torch.empty(0, dtype=torch.float32))

        self.weight_shape = (out_features, in_features)
        self.pad_len = 0

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.register_parameter('bias', None)

        self.is_quantized = False

        # [Optimization] 预计算解包表 (256 -> 2 values)
        # byte (uint8) -> (high 4-bit value, low 4-bit value)
        self.register_buffer('unpack_table', self._create_unpack_table())

    def _create_unpack_table(self):
        """创建一个 256x2 的查找表，将 uint8 映射为两个 4-bit 索引"""
        table = torch.zeros((256, 2), dtype=torch.long)
        for i in range(256):
            table[i, 0] = (i >> 4) & 0xF  # High
            table[i, 1] = i & 0xF  # Low
        return table

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 64, compute_dtype=torch.float32):
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            block_size=block_size,
            compute_dtype=compute_dtype
        )
        if linear.bias is not None:
            q_linear.bias.data = linear.bias.data.to(compute_dtype)

        with torch.no_grad():
            packed, absmax, shape, pad = quantize_blockwise(linear.weight.data, block_size=block_size)
            q_linear.weight_packed = packed
            q_linear.weight_absmax = absmax
            q_linear.weight_shape = shape
            q_linear.pad_len = pad
            q_linear.is_quantized = True

        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_quantized:
            raise RuntimeError("Linear4bit layer is not quantized yet!")

        # 1. 快速解包 (Table Lookup)
        # weight_packed: (N/2,) uint8 -> (N/2, 2) indices
        # 这一步在 CPU 上比手动位运算快得多
        indices = self.unpack_table[self.weight_packed.long()].view(-1)  # (N,)

        # 2. NF4 查表
        codebook = get_nf4_codebook(self.weight_packed.device)
        values = codebook[indices]  # (N,)

        # 3. 恢复尺度 (Rescale)
        # values: (Num_Blocks * 64)
        # absmax: (Num_Blocks)
        values = values.view(-1, self.block_size)
        absmax = self.weight_absmax.view(-1, 1)
        weight = values * absmax

        weight = weight.flatten()
        if self.pad_len > 0:
            weight = weight[:-self.pad_len]

        weight = weight.view(self.weight_shape).to(self.compute_dtype)

        # 4. Linear Calc
        x = x.to(self.compute_dtype)
        return F.linear(x, weight, self.bias)

# END OF FILE: finetune/peft/qlora/linear4bit.py