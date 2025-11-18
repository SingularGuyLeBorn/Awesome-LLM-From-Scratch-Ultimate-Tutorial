# FILE: models/blocks/attention/standard.py
"""
【流派一：标准与潜变量注意力 v2.0 - 真·MLA 推理】
- MLA 的 forward 方法已重构，以支持真正的 LatentKVCache。
- 在推理时，MLA 不再缓存展开的 KV，而是缓存低秩潜变量，并实时解压。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

try:
    from ..normalization.normalization import RMSNorm
    from inference.engine.kv_cache import LatentKVCache
except (ImportError, ValueError):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parents[3]))
    from models.blocks.normalization.normalization import RMSNorm
    from inference.engine.kv_cache import LatentKVCache


class StandardAttention(nn.Module):
    # ... (代码不变)
    def __init__(self, args):
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


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args):
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
        # 推理模式下，如果提供了 kv_cache，则走 latent cache 逻辑
        if kv_cache is not None:
            assert isinstance(kv_cache, LatentKVCache), "MLA requires a LatentKVCache for inference."
            return self._forward_inference(x, rope, layer_idx, kv_cache, start_pos)

        # 训练模式 或 PagedAttention 模式
        # (Paged Attention for MLA is complex and not implemented here; this will crash if attempted)
        if paged_attention_inputs is not None:
            raise NotImplementedError("PagedAttention for MLA is not supported in this version.")

        bs, seq_len, _ = x.shape

        # Query
        if self.q_lora_rank > 0:
            q_compressed = self.wq_down(x)
            q_compressed = self.q_norm(q_compressed)
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        # KV
        kv_compressed = self.wkv_down(x)
        kv_compressed = self.kv_norm(kv_compressed)
        kv_up = self.wkv_up(kv_compressed)
        kv_up = kv_up.view(bs, seq_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)
        k_rope_shared = self.wk_rope(x).view(bs, seq_len, 1, self.rope_head_dim)

        # RoPE
        q_pe = rope.apply_rotary_emb(q_pe)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        # Combine
        k_rope = k_rope_shared.expand(-1, -1, self.n_heads, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Transpose
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        if seq_len > 1:
            scores = scores + self.mask[:, :, :seq_len, :seq_len]

        probs = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

    def _forward_inference(self, x, rope, layer_idx, kv_cache: LatentKVCache, start_pos: int):
        bs, seq_len, _ = x.shape

        # 1. 计算当前 token 的 Q, c_KV, k_rope
        # Query
        if self.q_lora_rank > 0:
            q_compressed = self.q_norm(self.wq_down(x))
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        # KV Latent
        kv_compressed = self.kv_norm(self.wkv_down(x))
        # RoPE Key (shared, 注意这里没有 n_heads 维度)
        k_rope_shared = self.wk_rope(x).view(bs, seq_len, self.rope_head_dim)

        # 2. 应用 RoPE
        q_pe = rope.apply_rotary_emb(q_pe)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        # 3. 更新并获取完整的 Latent Cache
        full_c_kv, full_k_rope = kv_cache.update(layer_idx, start_pos, kv_compressed, k_rope_shared)

        # 4. 实时解压历史 KV
        # full_c_kv: (bs, full_seq_len, kv_lora_rank)
        # full_k_rope: (bs, full_seq_len, rope_head_dim)
        kv_up_history = self.wkv_up(full_c_kv)
        kv_up_history = kv_up_history.view(bs, -1, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope_history, v_history = torch.split(kv_up_history, [self.nope_head_dim, self.v_head_dim], dim=-1)

        # 5. 组合完整的 K
        # 广播 shared RoPE Key
        k_rope_history = full_k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        k_history = torch.cat([k_nope_history, k_rope_history], dim=-1)

        # 6. 组合完整的 Q
        q = torch.cat([q_nope, q_pe], dim=-1)

        # 7. Attention 计算
        q = q.transpose(1, 2)  # (bs, n_heads, seq_len, dim)
        k_history = k_history.transpose(1, 2)  # (bs, n_heads, full_seq_len, dim)
        v_history = v_history.transpose(1, 2)  # (bs, n_heads, full_seq_len, v_dim)

        scores = torch.matmul(q, k_history.transpose(2, 3)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        # Masking is not needed for single token generation (seq_len=1)

        probs = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(probs, v_history)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

# END OF FILE: models/blocks/attention/standard.py