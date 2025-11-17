# FILE: pretrain/data_loader.py
"""
【v2.3 - DDP 支持】为预训练阶段创建高效的数据加载器。
- 新增 ddp_rank 和 ddp_world_size 参数以支持 DistributedSampler。
"""
import sys
import os
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset, DataLoader
# [核心新增]
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from typing import Optional, Tuple

project_root = str(Path(os.path.dirname(__file__)).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PackedDataset(Dataset):
    """
    智能地将多个文档打包到一个序列中的数据集。
    支持只加载和索引前N个文档。
    """

    def __init__(self, bin_file: Path, block_size: int, eos_token_id: int, data_limit: Optional[int] = None):
        super().__init__()
        self.block_size = block_size
        self.eos_token_id = eos_token_id

        logging.info(f"开始处理打包数据集: {bin_file}")
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')

        logging.info("正在建立文档索引...")
        all_doc_boundaries = np.where(self.data == self.eos_token_id)[0]

        if data_limit is not None and data_limit > 0 and data_limit < len(all_doc_boundaries):
            self.doc_boundaries = all_doc_boundaries[:data_limit]
            logging.info(f"数据量限制已应用，将只使用前 {data_limit:,} 个文档。")
        else:
            self.doc_boundaries = all_doc_boundaries

        self.doc_boundaries = np.insert(self.doc_boundaries, 0, -1)
        logging.info(f"找到并索引了 {len(self.doc_boundaries) - 1:,} 个文档。")

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
            if space_left <= 0:
                break

            len_to_copy = min(doc_len, space_left)

            x[current_pos: current_pos + len_to_copy] = self.data[start: start + len_to_copy]
            y[current_pos: current_pos + len_to_copy - 1] = self.data[start + 1: start + len_to_copy]
            loss_mask[current_pos: current_pos + len_to_copy - 1] = 1.0

            current_pos += doc_len
            current_doc_idx += 1

        y[y == self.eos_token_id] = -1

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(loss_mask)


def get_pretrain_loaders(
        tokenizer_name: str,
        data_dir: Path,
        block_size: int,
        batch_size: int,
        num_workers: int = 0,
        train_data_limit: Optional[int] = None,
        val_data_limit: Optional[int] = None,
        return_train_loader: bool = True,
        return_val_loader: bool = True,
        ddp_rank: int = 0,
        ddp_world_size: int = 1
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    logging.info("开始准备 Packed-Sequences 数据加载器...")

    tokenizer_path = Path(tokenizer_name)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_name}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if ddp_rank == 0:
        logging.info(f"成功加载分词器，词表大小: {tokenizer.get_vocab_size()}, EOS ID: {eos_token_id}")

    train_loader, val_loader = None, None
    train_sampler, val_sampler = None, None

    if return_train_loader:
        train_bin_file = data_dir / "train.bin"
        if not train_bin_file.exists():
            raise FileNotFoundError(f"train.bin not found at {train_bin_file}. Please run data processing scripts.")
        train_dataset = PackedDataset(train_bin_file, block_size, eos_token_id, data_limit=train_data_limit)

        if ddp_world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # 只有在非DDP模式下才启用shuffle
            num_workers=num_workers,
            pin_memory=True
        )
        if ddp_rank == 0:
            logging.info(f"  -> 训练数据加载器已创建 (文档数: {len(train_dataset):,}, 批次数: {len(train_loader)})")
            if train_sampler: logging.info("     - 使用 DistributedSampler")

    if return_val_loader:
        val_bin_file = data_dir / "val.bin"
        if not val_bin_file.exists():
            raise FileNotFoundError(f"val.bin not found at {val_bin_file}. Please run data processing scripts.")
        val_dataset = PackedDataset(val_bin_file, block_size, eos_token_id, data_limit=val_data_limit)

        if ddp_world_size > 1:
            val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,  # 验证集通常不shuffle
            num_workers=num_workers,
            pin_memory=True
        )
        if ddp_rank == 0:
            logging.info(f"  -> 验证数据加载器已创建 (文档数: {len(val_dataset):,}, 批次数: {len(val_loader)})")
            if val_sampler: logging.info("     - 使用 DistributedSampler")

    if ddp_rank == 0:
        logging.info("Packed-Sequences 数据加载器准备完毕。")
    return train_loader, val_loader


# --- 单元测试代码 ---
if __name__ == "__main__":
    print("--- 单元测试: Packed-Sequences DataLoader ---")

    # 配置测试环境
    data_dir = project_root / "data_pipeline" / "processed_data"
    tokenizer_name = data_dir / "tinystories_project_vs4096.json"

    if not (data_dir / "train.bin").exists() or not (data_dir / "val.bin").exists():
        print("错误: train.bin 或 val.bin 不存在。请先运行 'data_pipeline/processing/build_pretrain_bins.py'")
        sys.exit(1)

    # --- 测试场景 1: 获取训练和验证加载器 ---
    print("\n--- 场景 1: 获取训练和验证加载器 (带数据限制) ---")
    train_loader, val_loader = get_pretrain_loaders(
        tokenizer_name=str(tokenizer_name),
        data_dir=data_dir,
        block_size=128,
        batch_size=4,
        train_data_limit=1000,
        val_data_limit=100
    )
    assert train_loader is not None and val_loader is not None
    print(f"训练加载器大小: {len(train_loader)} batches")
    print(f"验证加载器大小: {len(val_loader)} batches")
    x, y, mask = next(iter(train_loader))
    assert x.shape == (4, 128) and y.shape == (4, 128) and mask.shape == (4, 128)
    print("✅ 场景 1 通过！")

    # --- 测试场景 2: 只获取验证加载器 ---
    print("\n--- 场景 2: 只获取验证加载器 ---")
    train_loader_none, val_loader_only = get_pretrain_loaders(
        tokenizer_name=str(tokenizer_name),
        data_dir=data_dir,
        block_size=128,
        batch_size=4,
        return_train_loader=False,
        val_data_limit=200
    )
    assert train_loader_none is None and val_loader_only is not None
    print(f"返回的训练加载器: {train_loader_none}")
    print(f"返回的验证加载器大小: {len(val_loader_only)} batches")
    print("✅ 场景 2 通过！")

    # --- 测试场景 3: 只获取训练加载器 ---
    print("\n--- 场景 3: 只获取训练加载器 ---")
    train_loader_only, val_loader_none = get_pretrain_loaders(
        tokenizer_name=str(tokenizer_name),
        data_dir=data_dir,
        block_size=128,
        batch_size=4,
        return_val_loader=False,
        train_data_limit=500
    )
    assert train_loader_only is not None and val_loader_none is None
    print(f"返回的训练加载器大小: {len(train_loader_only)} batches")
    print(f"返回的验证加载器: {val_loader_none}")
    print("✅ 场景 3 通过！")

    print("\n🎉 DataLoader 单元测试全部成功！")
# END OF FILE: pretrain/data_loader.py