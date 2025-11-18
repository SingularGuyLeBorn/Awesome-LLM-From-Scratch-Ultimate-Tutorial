# FILE: inference/engine/kv_cache.py
# -*- coding: utf-8 -*-
"""
[v2.1 - 职责净化] KV 缓存 (Key-Value Cache) 的实现。

此文件现在只包含原始的、基于连续内存的 `KVCache`。
这个实现简单、直观，被 `inference/generate.py` 和 `inference/chat.py`
等基础推理脚本所使用。

更高级的、用于模拟PagedAttention的非连续内存管理，将在
`scheduler.py`, `paged_engine.py` 和 `block_manager.py` 中体现。
"""
import torch
from typing import Tuple


class KVCache:
    """
    一个用于存储多层Transformer的Key和Value状态的缓存（基于连续内存）。
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
# END OF FILE: inference/engine/kv_cache.py