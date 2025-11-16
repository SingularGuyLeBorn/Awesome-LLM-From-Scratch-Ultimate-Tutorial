# FILE: finetune/sft_data_loader.py
"""
[v2.2 - 索引优化版] SFT 专用数据加载器
- get_sft_loaders 现在可以按需提供验证集加载器。
- SFTDataset 现在会优先使用 .idx.npy 索引文件来加速初始化。
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tokenizers import Tokenizer
import logging
from pathlib import Path
from typing import Tuple, Optional

# 修复路径问题
import sys

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class SFTDataset(Dataset):
    def __init__(self, bin_file: Path, block_size: int, im_end_id: int, eos_token_id: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.im_end_id = im_end_id
        self.eos_token_id = eos_token_id

        # [核心修改] 优先加载索引文件
        index_file = bin_file.with_suffix('.idx.npy')
        if index_file.exists():
            logging.info(f"SFTDataset: 正在从索引文件 '{index_file.name}' 加载记录边界...")
            self.boundaries = np.load(index_file)
        else:
            logging.warning(f"SFTDataset: 索引文件 '{index_file.name}' 未找到。将实时扫描 .bin 文件（可能较慢）。")
            logging.warning("请重新运行 'data_pipeline/processing/build_sft_bins.py' 以生成索引。")
            self.boundaries = np.where(self.data == self.eos_token_id)[0]

        logging.info(f"  -> 找到 {len(self.boundaries):,} 条SFT记录。")

    def __len__(self):
        return len(self.boundaries)

    def __getitem__(self, idx):
        start_idx = self.boundaries[idx - 1] + 1 if idx > 0 else 0
        end_idx = self.boundaries[idx] + 1
        record_tokens = self.data[start_idx:end_idx].astype(np.int64)

        x = torch.full((self.block_size,), self.eos_token_id, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)

        seq_len = min(len(record_tokens), self.block_size)
        x[:seq_len] = torch.from_numpy(record_tokens[:seq_len])
        y[:seq_len - 1] = x[1:seq_len]

        x_tensor = x[:seq_len]
        im_end_indices = torch.where(x_tensor == self.im_end_id)[0]
        loss_mask = torch.zeros(self.block_size, dtype=torch.float)

        if len(im_end_indices) > 0:
            response_start_index = im_end_indices[0] + 1
            loss_mask[response_start_index: seq_len - 1] = 1.0

        y[loss_mask == 0] = -1
        return x, y, loss_mask


def get_sft_loaders(
        tokenizer_path: Path,
        sft_bin_file: Path,
        block_size: int,
        batch_size: int,
        num_workers: int = 0,
        provide_val_loader: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    print("\n--- 2. 准备SFT数据加载器 ---")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")

    if im_end_id is None or eos_token_id is None:
        raise ValueError("分词器缺少 <|im_end|> 或 <|endoftext|> token。")

    dataset = SFTDataset(sft_bin_file, block_size, im_end_id, eos_token_id)

    if len(dataset) == 0:
        raise ValueError("SFT 数据集为空，无法创建 DataLoader。请检查 sft_data.bin 文件是否正确生成。")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = None

    if provide_val_loader:
        # 简单起见，我们用同一个数据集作为验证集，但不打乱
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        print(f"✅ SFT 数据加载器准备完毕 (Train: {len(train_loader)} batches, Val: {len(val_loader)} batches)。")
    else:
        print(f"✅ SFT 训练数据加载器准备完毕 (共 {len(train_loader)} batches)。")

    return train_loader, val_loader

# END OF FILE: finetune/sft_data_loader.py