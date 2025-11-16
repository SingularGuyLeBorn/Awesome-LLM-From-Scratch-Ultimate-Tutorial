# FILE: data_pipeline/datasets.py
"""
[新增] 统一的数据集管理模块。
包含 PackedDataset 和 SFTDataset。
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from pathlib import Path
from typing import Optional


class PackedDataset(Dataset):
    """
    智能地将多个文档打包到一个序列中的数据集。
    支持只加载和索引前N个文档。
    """

    def __init__(self, bin_file: Path, block_size: int, eos_token_id: int, data_limit: Optional[int] = None):
        super().__init__()
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')

        logging.info(f"PackedDataset: 正在为 '{bin_file.name}' 建立文档索引...")
        all_doc_boundaries = np.where(self.data == self.eos_token_id)[0]

        if data_limit is not None and data_limit < len(all_doc_boundaries):
            self.doc_boundaries = all_doc_boundaries[:data_limit]
            logging.info(f"  -> 数据量限制已应用，将只使用前 {data_limit:,} 个文档。")
        else:
            self.doc_boundaries = all_doc_boundaries

        self.doc_boundaries = np.insert(self.doc_boundaries, 0, -1)
        logging.info(f"  -> 找到并索引了 {len(self.doc_boundaries) - 1:,} 个文档。")

    def __len__(self):
        return len(self.doc_boundaries) - 1

    def __getitem__(self, idx):
        doc_start_idx = self.doc_boundaries[idx] + 1
        x = np.full(self.block_size, self.eos_token_id, dtype=np.int64)
        y = np.full(self.block_size, -1, dtype=np.int64)
        loss_mask = np.zeros(self.block_size, dtype=np.float32)

        current_pos = 0
        current_doc_idx = idx
        while current_pos < self.block_size and current_doc_idx < len(self.doc_boundaries) - 1:
            start = self.doc_boundaries[current_doc_idx] + 1
            end = self.doc_boundaries[current_doc_idx + 1] + 1
            doc_len = end - start
            space_left = self.block_size - current_pos
            if space_left <= 0: break
            len_to_copy = min(doc_len, space_left)
            x[current_pos: current_pos + len_to_copy] = self.data[start: start + len_to_copy]
            y[current_pos: current_pos + len_to_copy - 1] = self.data[start + 1: start + len_to_copy]
            loss_mask[current_pos: current_pos + len_to_copy - 1] = 1.0
            current_pos += doc_len
            current_doc_idx += 1

        y[y == self.eos_token_id] = -1
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    """SFT专用数据集，核心功能是创建loss_mask。"""

    def __init__(self, bin_file: Path, block_size: int, im_end_id: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.im_end_id = im_end_id

        self.samples = []
        for i in range(0, len(self.data) - block_size, block_size):
            self.samples.append(i)
        logging.info(f"SFTDataset: 从 '{bin_file.name}' 加载了 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        chunk = self.data[start_idx: start_idx + self.block_size + 1]

        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        im_end_indices = torch.where(x == self.im_end_id)[0]
        loss_mask = torch.zeros_like(x, dtype=torch.float)

        if len(im_end_indices) > 0:
            response_start_index = im_end_indices[0] + 1
            loss_mask[response_start_index:] = 1.0

        y[y == self.im_end_id] = -1
        y[loss_mask == 0] = -1

        return x, y, loss_mask
# END OF FILE: data_pipeline/datasets.py