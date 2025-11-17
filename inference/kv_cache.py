# FILE: inference/kv_cache.py
# -*- coding: utf-8 -*-
"""
[v1.1 - 类型修复版] KV 缓存 (Key-Value Cache) 的实现。
- 修复了缓存数据类型被硬编码为 float16 的问题。
"""
import torch
from typing import Tuple, List


class KVCache:
    """
    一个用于存储多层Transformer的Key和Value状态的缓存。
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_layers: int,
                 n_kv_heads: int, head_dim: int, device: torch.device,
                 dtype: torch.dtype):  # [核心修改] 新增 dtype 参数
        """
        预先分配缓存所需的内存。

        Args:
            ...
            dtype: 缓存张量的数据类型, 应与模型参数类型一致。
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        # [核心修复] 使用传入的 dtype，而不是硬编码
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
        """
        用新的Key和Value更新缓存，并返回完整的序列。

        Args:
            layer_idx: 当前Transformer层的索引。
            start_pos: 当前新token在序列中的起始位置。
            xk: 当前计算出的Key张量, 形状 (bsz, n_kv_heads, seq_len, head_dim)
            xv: 当前计算出的Value张量, 形状 (bsz, n_kv_heads, seq_len, head_dim)

        Returns:
            一个元组 (keys, values)，包含从开始到现在的完整Key和Value序列。
        """
        bsz, _, seq_len, _ = xk.shape

        # 将新的 k/v 插入到缓存的正确位置
        self.cache_k[layer_idx, :bsz, :, start_pos: start_pos + seq_len, :] = xk
        self.cache_v[layer_idx, :bsz, :, start_pos: start_pos + seq_len, :] = xv

        # 从缓存中取出完整的序列
        # 我们只取出到当前生成位置 (start_pos + seq_len) 的数据
        keys = self.cache_k[layer_idx, :bsz, :, : start_pos + seq_len, :]
        values = self.cache_v[layer_idx, :bsz, :, : start_pos + seq_len, :]

        return keys, values
# END OF FILE: inference/kv_cache.py