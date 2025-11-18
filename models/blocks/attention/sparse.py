# FILE: models/blocks/attention/sparse.py
"""
【流派三：稀疏注意力 - 终极完整版 v3.5】
- 修复 'view size is not compatible...' 错误。
- 将所有敏感的 view 操作替换为 reshape。
- 增强内存连续性处理。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseAttentionBase(nn.Module):
    """稀疏注意力的基类，包含通用的投影和 GQA 处理逻辑"""

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = args.dropout
        self.resid_dropout = nn.Dropout(args.dropout)

        self.register_buffer("neg_inf", torch.tensor(-1e4))

    def _create_qkv_proj(self):
        """创建标准的 QKV 投影层"""
        return (
            nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False),
            nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False),
            nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        )

    def _safe_softmax(self, scores):
        """数值稳定的 Softmax"""
        return F.softmax(scores.float(), dim=-1).type_as(scores)

    def _apply_rotary(self, t, rope):
        """应用 RoPE"""
        return rope.apply_rotary_emb(t)

    def _get_qkv(self, x, wq, wk, wv, rope):
        bs, seq_len, _ = x.shape
        q = wq(x).view(bs, seq_len, self.n_heads, self.head_dim)
        k = wk(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        v = wv(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        q = self._apply_rotary(q, rope)
        k = self._apply_rotary(k, rope)

        # [核心修复] 加上 contiguous() 以确保内存布局连续，防止后续 view 报错
        return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()


class MixtureOfBlockAttention(SparseAttentionBase):
    def __init__(self, args):
        super().__init__(args)
        self.block_size = args.moba_block_size
        self.topk = args.moba_topk
        self.wq, self.wk, self.wv = self._create_qkv_proj()
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, **kwargs):
        bs, seq_len, _ = x.shape
        full_seq_len = start_pos + seq_len

        q, k, v = self._get_qkv(x, self.wq, self.wk, self.wv, rope)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, start_pos, k, v)

        if self.n_rep > 1:
            k_expanded = k.repeat_interleave(self.n_rep, dim=1)
            v_expanded = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_expanded, v_expanded = k, v

        if full_seq_len < self.block_size * 2:
            return self._standard_attention(q, k_expanded, v_expanded, start_pos)

        num_blocks = math.ceil(full_seq_len / self.block_size)
        pad_len = num_blocks * self.block_size - full_seq_len

        k_padded = F.pad(k_expanded, (0, 0, 0, pad_len))
        # [修复] 使用 reshape
        k_blocks = k_padded.reshape(bs, self.n_heads, num_blocks, self.block_size, self.head_dim)
        block_means = k_blocks.mean(dim=3)

        gating_scores = torch.matmul(q, block_means.transpose(-2, -1)) / math.sqrt(self.head_dim)

        q_block_idx = torch.arange(start_pos, full_seq_len, device=x.device) // self.block_size
        k_block_idx = torch.arange(num_blocks, device=x.device)
        block_mask = q_block_idx.unsqueeze(1) < k_block_idx.unsqueeze(0)
        gating_scores.masked_fill_(block_mask.unsqueeze(0).unsqueeze(0), self.neg_inf)

        curr_k = min(self.topk, num_blocks)
        _, selected_indices = torch.topk(gating_scores, k=curr_k, dim=-1)

        sparse_block_mask = torch.full_like(gating_scores, self.neg_inf)
        sparse_block_mask.scatter_(-1, selected_indices, 0.0)

        token_mask = sparse_block_mask.repeat_interleave(self.block_size, dim=-1)
        token_mask = token_mask[:, :, :, :full_seq_len]

        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(torch.ones(seq_len, full_seq_len, device=x.device), diagonal=start_pos + 1).bool()
        final_mask = torch.max(token_mask, causal_mask.unsqueeze(0).unsqueeze(0).float() * self.neg_inf)
        scores += final_mask

        probs = self._safe_softmax(scores)
        output = torch.matmul(probs, v_expanded)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

    def _standard_attention(self, q, k, v, start_pos):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        L, S = q.size(2), k.size(2)
        mask = torch.triu(torch.ones(L, S, device=q.device), diagonal=start_pos - S + L + 1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), self.neg_inf)

        probs = self._safe_softmax(scores)
        output = torch.matmul(probs, v)
        return self.resid_dropout(self.wo(output.transpose(1, 2).contiguous().view(q.size(0), L, -1)))


class NativeSparseAttention(SparseAttentionBase):
    def __init__(self, args):
        super().__init__(args)

        self.comp_block_size = args.nsa_compression_block_size
        self.sel_block_size = args.nsa_selection_block_size
        self.top_k_blocks = args.nsa_selected_blocks
        self.sliding_window = args.nsa_sliding_window_size

        self.wq_win, self.wk_win, self.wv_win = self._create_qkv_proj()
        self.wq_sel, self.wk_sel, self.wv_sel = self._create_qkv_proj()
        self.wq_comp, self.wk_comp, self.wv_comp = self._create_qkv_proj()

        self.compression_mlp = nn.Linear(self.head_dim * self.comp_block_size, self.head_dim, bias=False)
        self.gating_mlp = nn.Linear(self.dim, 3, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, **kwargs):
        bs, seq_len, _ = x.shape
        full_seq_len = start_pos + seq_len

        # --- 1. Prepare Q, K, V for all branches ---
        q_win, k_win, v_win = self._get_qkv(x, self.wq_win, self.wk_win, self.wv_win, rope)
        q_sel, k_sel, v_sel = self._get_qkv(x, self.wq_sel, self.wk_sel, self.wv_sel, rope)
        q_comp, k_comp, v_comp = self._get_qkv(x, self.wq_comp, self.wk_comp, self.wv_comp, rope)

        # --- 2. Sliding Window Attention ---
        k_win_sliced = k_win[:, :, max(0, seq_len - self.sliding_window):, :]
        v_win_sliced = v_win[:, :, max(0, seq_len - self.sliding_window):, :]
        win_start_pos = max(0, full_seq_len - self.sliding_window)
        out_win = self._compute_attention(q_win, k_win_sliced, v_win_sliced, q_start=start_pos, k_start=win_start_pos)

        # --- 3. Compression Attention ---
        num_blocks = math.ceil(seq_len / self.comp_block_size)
        pad_len = num_blocks * self.comp_block_size - seq_len

        k_comp_pad = F.pad(k_comp, (0, 0, 0, pad_len))
        v_comp_pad = F.pad(v_comp, (0, 0, 0, pad_len))

        # [核心修复] 使用 reshape 替代 view
        k_comp_blocks = k_comp_pad.reshape(bs, self.n_kv_heads, num_blocks, -1)
        v_comp_blocks = v_comp_pad.reshape(bs, self.n_kv_heads, num_blocks, -1)

        compressed_k = self.compression_mlp(k_comp_blocks)  # (bs, kv_heads, num_blocks, head_dim)
        compressed_v = self.compression_mlp(v_comp_blocks)

        out_comp = self._compute_attention(q_comp, compressed_k, compressed_v, is_causal=False)

        # --- 4. Selection Attention ---
        if self.n_rep > 1:
            gating_k = compressed_k.repeat_interleave(self.n_rep, dim=1)
        else:
            gating_k = compressed_k

        importance_scores = torch.einsum('bhqd,bhnd->bhqn', q_sel, gating_k).float()

        curr_k_blocks = min(self.top_k_blocks, num_blocks)
        _, topk_indices = torch.topk(importance_scores, k=curr_k_blocks, dim=-1)

        sparse_mask = torch.full_like(importance_scores, self.neg_inf)
        sparse_mask.scatter_(-1, topk_indices, 0.0)

        token_mask_sparse = sparse_mask.repeat_interleave(self.comp_block_size, dim=-1)
        token_mask_sparse = token_mask_sparse[:, :, :, :seq_len]

        out_sel = self._compute_attention(q_sel, k_sel, v_sel, attention_mask=token_mask_sparse, q_start=start_pos,
                                          k_start=start_pos)

        # --- 5. Gating and Fusion ---
        gates = self._safe_softmax(self.gating_mlp(x))
        g_win, g_comp, g_sel = gates.chunk(3, dim=-1)

        final_output = g_win * out_win + g_comp * out_comp + g_sel * out_sel

        return self.wo(self.resid_dropout(final_output))

    def _compute_attention(self, q, k, v, is_causal=True, q_start=0, k_start=0, attention_mask=None):
        bs, _, q_len, _ = q.shape
        k_len = k.size(2)

        if self.n_rep > 1 and k.size(1) != q.size(1):
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        if is_causal:
            pos_q = torch.arange(q_start, q_start + q_len, device=q.device)
            pos_k = torch.arange(k_start, k_start + k_len, device=q.device)
            causal_mask = pos_q.unsqueeze(1) < pos_k.unsqueeze(0)
            scores = scores.masked_fill(causal_mask, self.neg_inf)

        probs = self._safe_softmax(scores)
        output = torch.matmul(probs, v)
        return output.transpose(1, 2).contiguous().view(bs, q_len, -1)

# END OF FILE: models/blocks/attention/sparse.py