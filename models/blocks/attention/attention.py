# FILE: models/blocks/attention/attention.py
"""
【v2.2 - KV缓存集成版】
- forward 方法现在可以接收并处理 KVCache。
- 包含了完整的GQA和因果掩码的教学注释。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# 动态导入
try:
    from ..positional_encoding.positional_encoding import RoPE
    # 确保 KVCache 能被导入
    from inference.kv_cache import KVCache
except (ImportError, ValueError):
    # 兼容独立运行和项目内导入
    from models.blocks.positional_encoding.positional_encoding import RoPE
    from inference.kv_cache import KVCache


@dataclass
class AttentionConfig:
    dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256
    is_causal: bool = True


class Attention(nn.Module):
    def __init__(self, args: AttentionConfig):
        super().__init__()
        assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)

        self.is_causal = args.is_causal
        if self.is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
            self,
            x: torch.Tensor,
            rope: RoPE,
            # [核心修复] 新增KV缓存相关参数
            layer_idx: int,
            kv_cache: Optional[KVCache] = None,
            start_pos: int = 0,
            alibi_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape

        # 1. 线性投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. 调整形状
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)

        # 3. 应用RoPE
        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

        # 4. 转换为 (bs, n_heads, seq_len, head_dim) 以便计算
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 5. [核心逻辑] KV 缓存处理
        if kv_cache is not None:
            # 在推理时，更新缓存并获取完整的K,V序列
            keys, values = kv_cache.update(layer_idx, start_pos, xk, xv)
        else:
            # 在训练时，K,V就是当前计算出的xk, xv
            keys, values = xk, xv

        # 6. GQA: 扩展 K 和 V 头
        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        # 7. 计算注意力分数
        # 注意：这里的 `keys.shape[2]` 是包含缓存的完整序列长度
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 8. (可选) 应用 ALiBi 偏置
        if alibi_bias is not None:
            full_seq_len = keys.shape[2]
            scores = scores + alibi_bias[:, :, :full_seq_len, :full_seq_len]

        # 9. 应用因果掩码
        if self.is_causal:
            # 在推理的单token生成阶段 (seq_len=1)，这个掩码实际上只切片了一个很小的区域，
            # 但逻辑依然正确。在prompt预填充阶段，它能正确地应用掩码。
            current_seq_len = start_pos + seq_len
            scores = scores + self.mask[:, :, start_pos: current_seq_len, :current_seq_len]

        # 10. Softmax 和加权求和
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        # 11. 恢复形状并最终投影
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

# END OF FILE: models/blocks/attention/attention.py