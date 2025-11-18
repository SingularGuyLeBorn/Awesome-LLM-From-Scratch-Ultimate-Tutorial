# FILE: models/blocks/attention/attention.py
"""
【Attention Museum - v3.3 参数传递修复版】
- 修复了 "got multiple values for keyword argument 'paged_attention_inputs'" 错误。
- 统一了所有 Attention 实现的 forward 签名。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# 动态导入
try:
    from ..positional_encoding.positional_encoding import RoPE
    from inference.engine.kv_cache import KVCache
    from ..normalization.normalization import RMSNorm
except (ImportError, ValueError):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parents[3]))
    from models.blocks.positional_encoding.positional_encoding import RoPE
    from inference.engine.kv_cache import KVCache
    from models.blocks.normalization.normalization import RMSNorm
from models.config import ModelArgs


class Attention(nn.Module):
    """
    通用 Attention 包装器，根据配置分发到具体的实现。
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
        [核心修复] 显式接收 paged_attention_inputs，防止它留在 kwargs 中导致重复传递。
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


# =============================================================================
# 1. Standard Attention (MHA / GQA / MQA)
# =============================================================================
class StandardAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        if paged_attention_inputs is not None:
            return self._forward_paged(x, rope, layer_idx, paged_attention_inputs)

        bs, seq_len, _ = x.shape

        xq = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)

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

        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            scores = scores + self.mask[:, :, start_pos:current_seq_len, :current_seq_len]

        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, values)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

    def _forward_paged(self, x, rope, layer_idx, paged_inputs):
        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_inputs

        xq = self.wq(x).view(-1, self.n_heads, self.head_dim)
        xk = self.wk(x).view(-1, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(-1, self.n_kv_heads, self.head_dim)

        xq = rope.apply_rotary_emb_paged(xq, positions)
        xk = rope.apply_rotary_emb_paged(xk, positions)

        block_size = k_cache.shape[3]
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            current_ctx_len = context_lengths[seq_idx].item()
            start_pos = current_ctx_len - num_tokens

            for i in range(num_tokens):
                pos = start_pos + i
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size

                k_cache[block_idx, layer_idx, :, offset, :] = xk[token_idx]
                v_cache[block_idx, layer_idx, :, offset, :] = xv[token_idx]
                token_idx += 1

        output = torch.zeros_like(xq)
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            seq_len = context_lengths[seq_idx].item()

            gathered_k = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)
            gathered_v = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)

            for pos in range(seq_len):
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size
                gathered_k[:, pos, :] = k_cache[block_idx, layer_idx, :, offset, :]
                gathered_v[:, pos, :] = v_cache[block_idx, layer_idx, :, offset, :]

            if self.n_rep > 1:
                gathered_k = gathered_k.repeat_interleave(self.n_rep, dim=0)
                gathered_v = gathered_v.repeat_interleave(self.n_rep, dim=0)

            q_curr = xq[token_idx: token_idx + num_tokens].transpose(0, 1)

            scores = torch.matmul(q_curr, gathered_k.transpose(1, 2)) / math.sqrt(self.head_dim)

            if num_tokens > 1:
                q_pos = positions[token_idx: token_idx + num_tokens]
                k_pos = torch.arange(seq_len, device=x.device)
                mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

            probs = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, gathered_v)

            output[token_idx: token_idx + num_tokens] = attn_out.transpose(0, 1)
            token_idx += num_tokens

        output_flat = output.view(-1, self.n_heads * self.head_dim)
        return self.resid_dropout(self.wo(output_flat))


# =============================================================================
# 2. Multi-Head Latent Attention (MLA) - DeepSeek-V2
# =============================================================================
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.nope_head_dim = args.nope_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank > 0:
            self.wq_down = nn.Linear(args.dim, self.q_lora_rank, bias=False)
            self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.q_lora_rank, self.n_heads * self.rope_head_dim, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=args.norm_eps)
        else:
            self.wq_up = nn.Linear(args.dim, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(args.dim, self.n_heads * self.rope_head_dim, bias=False)

        self.wkv_down = nn.Linear(args.dim, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        self.wkv_up = nn.Linear(self.kv_lora_rank, self.n_heads * (self.nope_head_dim + self.v_head_dim), bias=False)
        self.wk_rope = nn.Linear(self.dim, self.rope_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, args.dim, bias=False)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        if paged_attention_inputs is not None:
            return self._forward_paged(x, rope, layer_idx, paged_attention_inputs)

        bs, seq_len, _ = x.shape

        if self.q_lora_rank > 0:
            q_compressed = self.wq_down(x)
            q_compressed = self.q_norm(q_compressed)
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        kv_compressed = self.wkv_down(x)
        kv_compressed = self.kv_norm(kv_compressed)
        kv_up = self.wkv_up(kv_compressed)
        kv_up = kv_up.view(bs, seq_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)
        k_rope_shared = self.wk_rope(x).view(bs, seq_len, 1, self.rope_head_dim)

        q_pe = rope.apply_rotary_emb(q_pe)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        k_rope = k_rope_shared.expand(-1, -1, self.n_heads, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, start_pos, k, v)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            scores = scores + self.mask[:, :, start_pos:current_seq_len, :current_seq_len]

        probs = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

    def _forward_paged(self, x, rope, layer_idx, paged_inputs):
        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_inputs

        if self.q_lora_rank > 0:
            q_compressed = self.wq_down(x)
            q_compressed = self.q_norm(q_compressed)
            q_nope = self.wq_up(q_compressed).view(-1, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(-1, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(-1, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(-1, self.n_heads, self.rope_head_dim)

        kv_compressed = self.wkv_down(x)
        kv_compressed = self.kv_norm(kv_compressed)
        kv_up = self.wkv_up(kv_compressed).view(-1, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)

        k_rope_shared = self.wk_rope(x).view(-1, 1, self.rope_head_dim)

        q_pe = rope.apply_rotary_emb_paged(q_pe, positions)
        k_rope_shared = rope.apply_rotary_emb_paged(k_rope_shared, positions)

        k_rope = k_rope_shared.expand(-1, self.n_heads, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        block_size = k_cache.shape[3]
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            current_ctx_len = context_lengths[seq_idx].item()
            start_pos = current_ctx_len - num_tokens

            for i in range(num_tokens):
                pos = start_pos + i
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size

                k_cache[block_idx, layer_idx, :, offset, :] = k[token_idx]
                v_cache[block_idx, layer_idx, :, offset, :] = v[token_idx]
                token_idx += 1

        output_v = torch.zeros(q.shape[0], self.n_heads, self.v_head_dim, device=x.device, dtype=x.dtype)

        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            seq_len = context_lengths[seq_idx].item()

            gathered_k = torch.zeros(self.n_heads, seq_len, self.nope_head_dim + self.rope_head_dim, device=x.device,
                                     dtype=x.dtype)
            gathered_v = torch.zeros(self.n_heads, seq_len, self.v_head_dim, device=x.device, dtype=x.dtype)

            for pos in range(seq_len):
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size
                gathered_k[:, pos, :] = k_cache[block_idx, layer_idx, :, offset, :]
                gathered_v[:, pos, :] = v_cache[block_idx, layer_idx, :, offset, :]

            q_curr = q[token_idx: token_idx + num_tokens].transpose(0, 1)

            scores = torch.matmul(q_curr, gathered_k.transpose(1, 2)) / math.sqrt(
                self.nope_head_dim + self.rope_head_dim)

            if num_tokens > 1:
                q_pos = positions[token_idx: token_idx + num_tokens]
                k_pos = torch.arange(seq_len, device=x.device)
                mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

            probs = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, gathered_v)

            output_v[token_idx: token_idx + num_tokens] = attn_out.transpose(0, 1)
            token_idx += num_tokens

        output_flat = output_v.view(-1, self.n_heads * self.v_head_dim)
        return self.wo(output_flat)


# =============================================================================
# 3. Linear Attention (ReLU Kernel)
# =============================================================================
class LinearAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def feature_map(self, x):
        return F.relu(x)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        if paged_attention_inputs is not None:
            raise NotImplementedError("PagedAttention not yet supported for Linear Attention variant.")

        bs, seq_len, _ = x.shape

        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = rope.apply_rotary_emb(q)
        k = rope.apply_rotary_emb(k)

        Q = self.feature_map(q)
        K = self.feature_map(k)

        attn = torch.matmul(Q, K.transpose(2, 3))
        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            attn = attn.masked_fill(self.mask[:, :, start_pos:current_seq_len, :current_seq_len] == float("-inf"), 0)

        denom = attn.sum(dim=-1, keepdim=True) + 1e-5
        attn = attn / denom

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))


# =============================================================================
# 4. Mixture of Block Attention (MoBA)
# =============================================================================
class MixtureOfBlockAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.impl = StandardAttention(args)  # Use standard as base
        self.block_size = args.moba_block_size
        self.topk = args.moba_topk
        self.gate = nn.Linear(args.dim, 1, bias=False)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        # Pass through
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