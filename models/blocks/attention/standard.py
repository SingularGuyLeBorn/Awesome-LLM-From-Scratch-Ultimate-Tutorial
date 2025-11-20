# FILE: models/blocks/attention/standard.py
"""
【流派一：标准与潜变量注意力 v2.3 - 推理位置编码修复版】
- 修复 MLA 推理优化中 RoPE 位置索引错误 (始终使用 pos 0 的 bug)。
- 使用 apply_rotary_emb_paged 确保 decode 阶段使用正确的绝对位置。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        # (B, L, D) -> (B, L, H, D)
        xq = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)

        # [Fix] Transpose to (B, H, L, D) for RoPE
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)  # XV doesn't need RoPE, but we transpose for consistency/cache

        # Apply RoPE
        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

        # Cache Update: expects (B, H, L, D)
        if kv_cache is not None:
            keys, values = kv_cache.update(layer_idx, start_pos, xk, xv)
        else:
            keys, values = xk, xv

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        # Attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            scores = scores + self.mask[:, :, start_pos:current_seq_len, :current_seq_len]

        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, values)

        # (B, H, L, D) -> (B, L, H, D) -> (B, L, D)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

    def _forward_paged(self, x, rope, layer_idx, paged_inputs):
        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_inputs

        # Paged input x is usually (Total_Tokens, Dim)
        # View as (Total_Tokens, Heads, HeadDim)
        xq = self.wq(x).view(-1, self.n_heads, self.head_dim)
        xk = self.wk(x).view(-1, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(-1, self.n_kv_heads, self.head_dim)

        # Paged RoPE handles (Total_Tokens, Heads, HeadDim) correctly
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

        # Q Projections
        if self.q_lora_rank > 0:
            self.wq_down = nn.Linear(args.dim, self.q_lora_rank, bias=False)
            self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.q_lora_rank, self.n_heads * self.rope_head_dim, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=args.norm_eps)
        else:
            self.wq_up = nn.Linear(args.dim, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(args.dim, self.n_heads * self.rope_head_dim, bias=False)

        # KV Projections
        self.wkv_down = nn.Linear(args.dim, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)

        # [Optimization Key] wkv_up maps Latent -> Heads * (Nope + V)
        self.wkv_up = nn.Linear(self.kv_lora_rank, self.n_heads * (self.nope_head_dim + self.v_head_dim), bias=False)

        # RoPE Key (Decoupled, not compressed)
        self.wk_rope = nn.Linear(self.dim, self.rope_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, args.dim, bias=False)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        if kv_cache is not None:
            assert isinstance(kv_cache, LatentKVCache), "MLA requires a LatentKVCache for inference."
            return self._forward_inference_optimized(x, rope, layer_idx, kv_cache, start_pos)

        if paged_attention_inputs is not None:
            raise NotImplementedError("PagedAttention for MLA is not supported in this version.")

        # Training Path
        bs, seq_len, _ = x.shape

        # 1. Query Generation
        if self.q_lora_rank > 0:
            q_compressed = self.wq_down(x)
            q_compressed = self.q_norm(q_compressed)
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        # 2. KV Generation
        kv_compressed = self.wkv_down(x)
        kv_compressed = self.kv_norm(kv_compressed)

        # Decompress KV
        kv_up = self.wkv_up(kv_compressed)
        kv_up = kv_up.view(bs, seq_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)

        # k_rope_shared: (B, L, D) -> (B, L, 1, D)
        k_rope_shared = self.wk_rope(x).view(bs, seq_len, 1, self.rope_head_dim)

        # 3. Apply RoPE (Need Transpose to B, H, L, D)
        q_pe = q_pe.transpose(1, 2)  # (B, H, L, D)
        k_rope_shared = k_rope_shared.transpose(1, 2)  # (B, 1, L, D)

        q_pe = rope.apply_rotary_emb(q_pe)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        # Transpose back: (B, L, H, D)
        q_pe = q_pe.transpose(1, 2)
        k_rope_shared = k_rope_shared.transpose(1, 2)

        # 4. Combine Heads
        k_rope = k_rope_shared.expand(-1, -1, self.n_heads, -1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # 5. Attention (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # Transpose V

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        if seq_len > 1:
            scores = scores + self.mask[:, :, :seq_len, :seq_len]

        probs = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

    def _forward_inference_optimized(self, x, rope, layer_idx, kv_cache: LatentKVCache, start_pos: int):
        bs, seq_len, _ = x.shape
        assert seq_len == 1, "Inference optimized path assumes generating 1 token at a time."

        # --- 1. 生成当前 Token 的组件 ---

        # Query
        if self.q_lora_rank > 0:
            q_compressed = self.q_norm(self.wq_down(x))
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        # KV Latent
        kv_compressed = self.kv_norm(self.wkv_down(x))  # (bs, 1, kv_lora_rank)

        # RoPE Key (Current Token, Shared)
        # (B, L, D)
        k_rope_shared = self.wk_rope(x).view(bs, seq_len, self.rope_head_dim)

        # --- 2. 应用 RoPE (Fix: 使用 apply_rotary_emb_paged 并传入绝对位置) ---

        # 构造绝对位置索引
        # positions: (bs * seq_len) -> [start_pos, start_pos+1, ...]
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(bs, -1).flatten()

        # Apply RoPE to Query
        # q_pe: (B, L, H, D) -> View as (Tokens, H, D)
        q_pe_flat = q_pe.view(bs * seq_len, self.n_heads, self.rope_head_dim)
        q_pe_out = rope.apply_rotary_emb_paged(q_pe_flat, positions)
        # Reshape back to (B, H, L, D) for Attention (Transpose required)
        q_pe = q_pe_out.view(bs, seq_len, self.n_heads, self.rope_head_dim).transpose(1, 2)

        # Apply RoPE to Shared Key
        # k_rope_shared: (B, L, D) -> View as (Tokens, 1, D)
        k_rope_flat = k_rope_shared.view(bs * seq_len, 1, self.rope_head_dim)
        k_rope_out = rope.apply_rotary_emb_paged(k_rope_flat, positions)
        # Reshape back to (B, L, D)
        k_rope_shared = k_rope_out.view(bs, seq_len, self.rope_head_dim)

        # --- 3. 更新缓存 (存的是压缩状态!) ---
        # k_rope_shared is (bs, seq_len, rope_head_dim) matching cache expectation
        full_c_kv, full_k_rope = kv_cache.update(layer_idx, start_pos, kv_compressed, k_rope_shared)

        # --- 4. 计算 Attention Score (优化核心) ---

        # A. 计算 S_pe
        # q_pe: (bs, n_heads, 1, rope_dim)
        # k_rope_hist: (bs, full_len, rope_dim) -> broadcast -> (bs, 1, full_len, rope_dim)
        k_rope_hist_heads = full_k_rope.unsqueeze(1)  # (bs, 1, full_len, rope_dim)

        scores_pe = torch.matmul(q_pe, k_rope_hist_heads.transpose(-2, -1))

        # B. 计算 S_nope
        w_up_weight = self.wkv_up.weight
        head_dim_total = self.nope_head_dim + self.v_head_dim
        w_up_reshaped = w_up_weight.view(self.n_heads, head_dim_total, self.kv_lora_rank)

        w_uk = w_up_reshaped[:, :self.nope_head_dim, :]
        w_uv = w_up_reshaped[:, self.nope_head_dim:, :]

        # q_nope: (bs, L, H, D) -> Transpose -> (bs, H, L, D)
        q_nope_heads = q_nope.transpose(1, 2)

        # q_absorbed
        q_absorbed = torch.einsum('bhtd,hdr->bhtr', q_nope_heads, w_uk)

        # scores_nope
        scores_nope = torch.matmul(q_absorbed, full_c_kv.transpose(1, 2).unsqueeze(1))

        # Total Score
        scores = (scores_nope + scores_pe) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        probs = F.softmax(scores.float(), dim=-1).type_as(x)  # (bs, n_heads, 1, full_len)

        # --- 5. 计算 Output ---
        # Step 1: Latent Aggregate
        latent_output = torch.matmul(probs, full_c_kv.unsqueeze(1))

        # Step 2: Decompress
        output = torch.einsum('bhtr,hvr->bhtv', latent_output, w_uv)

        # --- 6. Final Projection ---
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

# END OF FILE: models/blocks/attention/standard.py