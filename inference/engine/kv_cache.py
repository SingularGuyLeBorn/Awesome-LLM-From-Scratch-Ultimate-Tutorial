# FILE: inference/engine/kv_cache.py
# -*- coding: utf-8 -*-
"""
【v3.0 - KV Cache 架构重构】
- 引入 KVCacheBase 基类。
- StandardKVCache: 用于 MHA/GQA/MQA 的传统 KV 缓存。
- LatentKVCache: 为 MLA (DeepSeek-V2) 实现的潜变量缓存，显著降低内存占用。
"""
import torch
from typing import Tuple


class KVCacheBase:
    """KV 缓存的抽象基类。"""

    def update(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class StandardKVCache(KVCacheBase):
    """
    用于 MHA/GQA/MQA 的标准 KV 缓存 (基于连续内存)。
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_layers: int,
                 n_kv_heads: int, head_dim: int, device: torch.device,
                 dtype: torch.dtype):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.cache_k = torch.zeros(
            (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        self.cache_v = torch.zeros(
            (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )

    def update(self, layer_idx: int, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        bsz, _, seq_len, _ = xk.shape
        self.cache_k[layer_idx, :bsz, :, start_pos: start_pos + seq_len, :] = xk
        self.cache_v[layer_idx, :bsz, :, start_pos: start_pos + seq_len, :] = xv
        keys = self.cache_k[layer_idx, :bsz, :, : start_pos + seq_len, :]
        values = self.cache_v[layer_idx, :bsz, :, : start_pos + seq_len, :]
        return keys, values


class LatentKVCache(KVCacheBase):
    """
    为 MLA (Multi-Head Latent Attention) 设计的潜变量缓存。
    核心思想：只缓存低秩的 c_KV 和解耦的 k_rope，从而大幅节省显存。
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_layers: int,
                 kv_lora_rank: int, rope_head_dim: int, device: torch.device,
                 dtype: torch.dtype):
        # 1. 缓存压缩后的潜变量 c_KV
        self.cache_latent = torch.zeros(
            (n_layers, max_batch_size, max_seq_len, kv_lora_rank),
            dtype=dtype, device=device
        )

        # 2. 缓存解耦后的 RoPE Key (shared across heads)
        self.cache_k_rope = torch.zeros(
            (n_layers, max_batch_size, max_seq_len, rope_head_dim),
            dtype=dtype, device=device
        )

    def update(self, layer_idx: int, start_pos: int, c_kv: torch.Tensor, k_rope: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        更新潜变量缓存。
        Args:
            c_kv: (bs, seq_len, kv_lora_rank)
            k_rope: (bs, seq_len, rope_head_dim)
        Returns:
            完整的 c_KV 和 k_rope 历史。
        """
        bsz, seq_len, _ = c_kv.shape
        self.cache_latent[layer_idx, :bsz, start_pos: start_pos + seq_len, :] = c_kv
        self.cache_k_rope[layer_idx, :bsz, start_pos: start_pos + seq_len, :] = k_rope

        full_c_kv = self.cache_latent[layer_idx, :bsz, :start_pos + seq_len, :]
        full_k_rope = self.cache_k_rope[layer_idx, :bsz, :start_pos + seq_len, :]

        return full_c_kv, full_k_rope
# END OF FILE: inference/engine/kv_cache.py