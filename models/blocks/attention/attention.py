# FILE: models/blocks/attention/attention.py
"""
【重构版】从零手写实现Transformer中的注意力机制。
- 模块内部自己管理因果Mask，实现解耦。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# 动态导入RoPE，以便此文件可以独立测试
try:
    from ..positional_encoding.positional_encoding import RoPE
except ImportError:
    from models.blocks.positional_encoding.positional_encoding import RoPE


@dataclass
class AttentionConfig:
    dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256
    is_causal: bool = True  # [新增]


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

        # [修改] 模块内部自己管理Mask
        self.is_causal = args.is_causal
        if self.is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
            self,
            x: torch.Tensor,
            rope: RoPE,
            alibi_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        if alibi_bias is not None:
            scores = scores + alibi_bias[:, :, :seq_len, :seq_len]

        # [修改] 使用内部管理的Mask
        if self.is_causal:
            scores = scores + self.mask[:, :, :seq_len, :seq_len]

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

# END OF FILE: models/blocks/attention/attention.py