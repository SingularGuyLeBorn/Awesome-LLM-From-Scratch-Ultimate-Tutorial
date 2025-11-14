# FILE: pretrain/data_loader.py
"""
【Packed Sequences版】为预训练阶段创建高效的数据加载器。
"""
import sys
import os
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

project_root = str(Path(os.path.dirname(__file__)).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PackedDataset(Dataset):
    """
    智能地将多个文档打包到一个序列中的数据集。
    """

    def __init__(self, bin_file: Path, block_size: int, eos_token_id: int):
        super().__init__()
        self.block_size = block_size
        self.eos_token_id = eos_token_id

        logging.info(f"开始处理打包数据集: {bin_file}")
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')

        # 查找所有文档的边界 (EOS token的位置)
        logging.info("正在建立文档索引...")
        self.doc_boundaries = np.where(self.data == self.eos_token_id)[0]
        # 在开头加入一个虚拟的边界
        self.doc_boundaries = np.insert(self.doc_boundaries, 0, -1)
        logging.info(f"找到 {len(self.doc_boundaries) - 1:,} 个文档。")

    def __len__(self):
        # 长度是文档的数量，因为我们从每个文档开始构建一个样本
        return len(self.doc_boundaries) - 1

    def __getitem__(self, idx):
        # 随机选择一个文档作为起点
        doc_start_idx = self.doc_boundaries[idx] + 1

        # 找到能填满 block_size 的文档结尾
        buffer = self.data[doc_start_idx: doc_start_idx + self.block_size + 1]

        # 准备输入 x, 目标 y, 和损失掩码 loss_mask
        x = np.full(self.block_size, 0, dtype=np.int64)
        y = np.full(self.block_size, 0, dtype=np.int64)
        loss_mask = np.zeros(self.block_size, dtype=np.float32)

        current_pos = 0
        current_doc_idx = idx

        # 循环打包文档
        while current_pos < self.block_size and current_doc_idx < len(self.doc_boundaries) - 1:
            start = self.doc_boundaries[current_doc_idx] + 1
            end = self.doc_boundaries[current_doc_idx + 1] + 1  # 包含EOS token
            doc_len = end - start

            # 检查剩余空间是否足够
            space_left = self.block_size - current_pos
            if space_left <= 0:
                break

            # 确定要复制的长度
            len_to_copy = min(doc_len, space_left)

            # 填充 x, y 和 loss_mask
            x[current_pos: current_pos + len_to_copy] = self.data[start: start + len_to_copy]
            # y 是 x 的移位版本
            y[current_pos: current_pos + len_to_copy - 1] = self.data[start + 1: start + len_to_copy]
            # loss_mask: EOS token的位置不计算损失
            loss_mask[current_pos: current_pos + len_to_copy - 1] = 1.0  # EOS 前的所有token都计算损失

            current_pos += doc_len
            current_doc_idx += 1

        # 如果最后一个token不是EOS，y的最后一个位置需要填充
        if x[self.block_size - 1] != self.eos_token_id:
            y[self.block_size - 1] = x[0]  # 循环预测，或者填充一个特殊值

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(loss_mask)


def get_loaders(
        tokenizer_name: str,
        data_dir: Path,
        block_size: int,
        batch_size: int,
        num_workers: int = 0
) -> (DataLoader, DataLoader):
    logging.info("开始准备 Packed-Sequences 数据加载器...")

    # 1. 加载分词器以获取eos_token_id
    tokenizer_path = Path(tokenizer_name)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_name}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    logging.info(f"成功加载分词器，词表大小: {tokenizer.get_vocab_size()}, EOS ID: {eos_token_id}")

    # 2. 准备数据集
    train_bin_file = data_dir / "train.bin"
    val_bin_file = data_dir / "val.bin"

    if not train_bin_file.exists() or not val_bin_file.exists():
        raise FileNotFoundError(
            "train.bin or val.bin not found. "
            "Please run 'data_pipeline/processing/process_tinystories.py' first."
        )

    train_dataset = PackedDataset(train_bin_file, block_size, eos_token_id)
    val_dataset = PackedDataset(val_bin_file, block_size, eos_token_id)

    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logging.info("Packed-Sequences 数据加载器准备完毕。")
    return train_loader, val_loader


# --- 测试代码 ---
if __name__ == "__main__":
    print("--- 测试 Packed-Sequences DataLoader ---")

    data_dir = project_root / "data_pipeline" / "processed_data"
    tokenizer_name = data_dir / "tinystories_project_vs4096.json"

    if not (data_dir / "train.bin").exists():
        print("错误: train.bin 不存在. 请先运行 process_tinystories.py")
        sys.exit(1)

    train_loader, val_loader = get_loaders(
        tokenizer_name=str(tokenizer_name),
        data_dir=data_dir,
        block_size=128,
        batch_size=4
    )

    print("\n--- 检查一批训练数据 ---")
    x, y, mask = next(iter(train_loader))

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"mask shape: {mask.shape}")

    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    assert mask.shape == (4, 128)

    print("\n✅ DataLoader 测试成功！")

# END OF FILE: pretrain/data_loader.py