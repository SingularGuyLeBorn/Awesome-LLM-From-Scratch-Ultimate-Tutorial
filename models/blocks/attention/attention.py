# FILE: models/blocks/attention/attention.py
"""
【v2.9 - 真正最终修复版】
- [核心修复] 恢复了在“终极诊断版”中被验证为正确的、基于绝对位置的
  因果掩码逻辑。之前版本在移除调试打印时，错误地将此关键修复回滚了。
  本次修复确保了 PagedAttention 在 Prefill 阶段的正确性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# 动态导入
try:
    from ..positional_encoding.positional_encoding import RoPE
    from inference.engine.kv_cache import KVCache
except (ImportError, ValueError):
    from models.blocks.positional_encoding.positional_encoding import RoPE
    from inference.engine.kv_cache import KVCache


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
            layer_idx: int,
            kv_cache: Optional[KVCache] = None,
            start_pos: int = 0,
            alibi_bias: Optional[torch.Tensor] = None,
            paged_attention_inputs: Optional[Tuple] = None
    ) -> torch.Tensor:

        if paged_attention_inputs is not None:
            return self._paged_attention_forward(x, rope, layer_idx, paged_attention_inputs)
        else:
            return self._standard_attention_forward(x, rope, layer_idx, kv_cache, start_pos, alibi_bias)

    def _standard_attention_forward(self, x, rope, layer_idx, kv_cache, start_pos, alibi_bias):
        bs, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if kv_cache is not None:
            keys, values = kv_cache.update(layer_idx, start_pos, xk, xv)
        else:
            keys, values = xk, xv

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if alibi_bias is not None:
            full_seq_len = keys.shape[2]
            scores = scores + alibi_bias[:, :, :full_seq_len, :full_seq_len]

        if self.is_causal:
            current_seq_len = start_pos + seq_len
            scores = scores + self.mask[:, :, start_pos: current_seq_len, :current_seq_len]

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

    def _paged_attention_forward(self, x, rope, layer_idx, paged_attention_inputs):
        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_attention_inputs

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(-1, self.n_heads, self.head_dim)
        xk = xk.view(-1, self.n_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_kv_heads, self.head_dim)

        xq = rope.apply_rotary_emb_paged(xq, positions)
        xk = rope.apply_rotary_emb_paged(xk, positions)

        block_size = k_cache.shape[3]
        token_idx_in_flat_batch = 0
        for seq_idx in range(len(tokens_per_seq)):
            num_tokens = tokens_per_seq[seq_idx].item()
            start_pos = context_lengths[seq_idx].item() - num_tokens

            for i in range(num_tokens):
                pos = start_pos + i
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size

                k_cache[block_idx, layer_idx, :, offset, :] = xk[token_idx_in_flat_batch, :, :]
                v_cache[block_idx, layer_idx, :, offset, :] = xv[token_idx_in_flat_batch, :, :]
                token_idx_in_flat_batch += 1

        output = torch.zeros_like(xq)
        token_idx_in_flat_batch = 0
        for seq_idx in range(len(tokens_per_seq)):
            num_tokens = tokens_per_seq[seq_idx].item()
            is_prefill = num_tokens > 1
            seq_len = context_lengths[seq_idx].item()

            gathered_k = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)
            gathered_v = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)

            for token_pos in range(seq_len):
                block_idx = block_tables[seq_idx, token_pos // block_size].item()
                offset = token_pos % block_size
                gathered_k[:, token_pos, :] = k_cache[block_idx, layer_idx, :, offset, :]
                gathered_v[:, token_pos, :] = v_cache[block_idx, layer_idx, :, offset, :]

            if self.n_rep > 1:
                gathered_k = gathered_k.repeat_interleave(self.n_rep, dim=0)
                gathered_v = gathered_v.repeat_interleave(self.n_rep, dim=0)

            q = xq[token_idx_in_flat_batch: token_idx_in_flat_batch + num_tokens].transpose(0, 1)

            scores = torch.matmul(q, gathered_k.transpose(1, 2)) / math.sqrt(self.head_dim)

            # [核心修复] 使用基于绝对位置的正确因果掩码逻辑
            if self.is_causal:
                if is_prefill:
                    q_positions = positions[token_idx_in_flat_batch: token_idx_in_flat_batch + num_tokens]
                    k_positions = torch.arange(0, seq_len, device=q.device)
                    mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
                    scores = scores.masked_fill(mask, float('-inf'))

            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            attn_out = torch.matmul(scores, gathered_v)

            output[token_idx_in_flat_batch: token_idx_in_flat_batch + num_tokens] = attn_out.transpose(0, 1)
            token_idx_in_flat_batch += num_tokens

        output_flat = output.view(-1, self.n_heads * self.head_dim)
        return self.resid_dropout(self.wo(output_flat))
# END OF FILE: models/blocks/attention/attention.py