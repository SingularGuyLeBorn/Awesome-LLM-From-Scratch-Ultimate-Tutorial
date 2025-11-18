# FILE: inference/engine/block_manager.py
# -*- coding: utf-8 -*-
"""
[新增] PagedAttention 的核心组件：块管理器 (Block Manager)。

BlockManager 负责管理一个物理的 KV 缓存池，将其划分为固定大小的块(block)，
并处理块的分配和释放。这是 vLLM 实现 PagedAttention 的核心思想。
"""
import torch
from typing import List, Dict


class BlockManager:
    def __init__(
            self,
            num_blocks: int,
            block_size: int,
            n_layers: int,
            n_kv_heads: int,
            head_dim: int,
            device: torch.device,
            dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # 1. 创建物理 KV 缓存池
        # 这是一个巨大的张量，包含了所有可用的内存
        self.k_cache_pool = torch.zeros(
            (num_blocks, n_layers, n_kv_heads, block_size, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache_pool = torch.zeros(
            (num_blocks, n_layers, n_kv_heads, block_size, head_dim),
            device=device,
            dtype=dtype,
        )

        # 2. 创建空闲块列表
        # `free_blocks` 是一个简单的栈，用于快速分配和回收块的物理索引
        self.free_blocks = list(range(num_blocks))

        # 3. 维护每个序列的块表
        # `block_tables` 是一个字典，key 是序列ID (seq_id)，value 是一个列表，
        # 存储该序列占用的物理块的索引。
        self.block_tables: Dict[int, List[int]] = {}

    def can_allocate(self, num_required_blocks: int) -> bool:
        """检查是否有足够的空闲块来分配。"""
        return len(self.free_blocks) >= num_required_blocks

    def allocate(self, seq_id: int, num_required_blocks: int):
        """为一个新序列分配块。"""
        if not self.can_allocate(num_required_blocks):
            raise ValueError(f"内存不足：需要 {num_required_blocks} 个块，但只有 {len(self.free_blocks)} 个可用。")

        block_indices = [self.free_blocks.pop() for _ in range(num_required_blocks)]
        self.block_tables[seq_id] = block_indices

    def free(self, seq_id: int):
        """释放一个序列占用的所有块。"""
        if seq_id not in self.block_tables:
            return

        block_indices = self.block_tables.pop(seq_id)
        self.free_blocks.extend(block_indices)

    def append_block(self, seq_id: int):
        """为一个现有序列追加一个新块。"""
        if not self.can_allocate(1):
            raise ValueError("内存不足：无法为序列追加新块。")

        if seq_id not in self.block_tables:
            raise ValueError(f"序列 {seq_id} 不存在，无法追加块。")

        block_index = self.free_blocks.pop()
        self.block_tables[seq_id].append(block_index)

    def get_block_table(self, seq_id: int) -> List[int]:
        """获取一个序列的块表。"""
        return self.block_tables.get(seq_id, [])

    def get_num_free_blocks(self) -> int:
        """获取当前空闲块的数量。"""
        return len(self.free_blocks)