# FILE: finetune/sft_data_loader.py
"""
[v2.0 - 核心逻辑重构版] SFT 专用数据加载器
- 修复了数据集长度为0的致命bug。
- 采用正确的“边界索引”逻辑，而不是错误的“切块”逻辑。
- 增强了__getitem__，使其能够处理可变长度的SFT记录。
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tokenizers import Tokenizer
import logging
from pathlib import Path

# 修复路径问题
import sys

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_pipeline.datasets import SFTDataset  # 我们将在这里重定义它


class SFTDataset(Dataset):
    def __init__(self, bin_file: Path, block_size: int, im_end_id: int, eos_token_id: int):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.im_end_id = im_end_id
        self.eos_token_id = eos_token_id

        # --- [核心修复] 采用“边界索引”逻辑 ---
        logging.info(f"SFTDataset: 正在为 '{bin_file.name}' 建立记录索引...")
        # 找到所有记录的结束位置
        self.boundaries = np.where(self.data == self.eos_token_id)[0]
        logging.info(f"  -> 找到 {len(self.boundaries):,} 条SFT记录。")

    def __len__(self):
        return len(self.boundaries)

    def __getitem__(self, idx):
        # 确定当前记录的边界
        start_idx = self.boundaries[idx - 1] + 1 if idx > 0 else 0
        end_idx = self.boundaries[idx] + 1

        # 提取完整的SFT记录 (prompt + response + special tokens)
        record_tokens = self.data[start_idx:end_idx].astype(np.int64)

        # 准备固定长度的张量
        x = torch.full((self.block_size,), self.eos_token_id, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)

        # 截断或填充
        seq_len = min(len(record_tokens), self.block_size)
        x[:seq_len] = torch.from_numpy(record_tokens[:seq_len])

        # y 是 x 的移位版本
        y[:seq_len - 1] = x[1:seq_len]

        # --- 核心逻辑：创建 loss_mask ---
        x_tensor = x[:seq_len]  # 只在有效长度内查找
        im_end_indices = torch.where(x_tensor == self.im_end_id)[0]

        loss_mask = torch.zeros(self.block_size, dtype=torch.float)

        if len(im_end_indices) > 0:
            # 只在第一个 <|im_end|> 之后的部分计算损失
            response_start_index = im_end_indices[0] + 1
            # 确保掩码只应用在有效token上
            loss_mask[response_start_index: seq_len - 1] = 1.0

        # 应用掩码到 y，所有不计算损失的位置都设为-1
        y[loss_mask == 0] = -1

        return x, y, loss_mask


def get_sft_loaders(
        tokenizer_path: Path,
        sft_bin_file: Path,
        block_size: int,
        batch_size: int,
        num_workers: int = 0
) -> (DataLoader, None):
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

    print(f"✅ SFT 数据加载器准备完毕 (共 {len(dataset)} 个样本)。")
    return train_loader, None
# END OF FILE: finetune/sft_data_loader.py