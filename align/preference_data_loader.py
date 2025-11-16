# FILE: align/preference_data_loader.py
"""
[v1.2 - Attention Mask版] 偏好数据 (Preference Data) 加载器
- 现在可以同时被 DPO 和 RM 训练脚本使用。
- __getitem__ 现在会返回 attention_mask 以支持更健壮的奖励模型。
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from tokenizers import Tokenizer


class PreferenceDataset(Dataset):
    def __init__(self, chosen_bin_file: Path, rejected_bin_file: Path, block_size: int, pad_token_id: int):
        self.block_size = block_size
        self.pad_token_id = pad_token_id

        # 使用只读模式加载 memory-mapped 文件，避免消耗大量内存
        chosen_data = np.memmap(chosen_bin_file, dtype=np.uint16, mode='r')
        rejected_data = np.memmap(rejected_bin_file, dtype=np.uint16, mode='r')

        # 将一维数据重塑为 (num_samples, block_size) 的二维数组
        self.chosen_samples = torch.from_numpy(chosen_data.reshape(-1, block_size).astype(np.int64))
        self.rejected_samples = torch.from_numpy(rejected_data.reshape(-1, block_size).astype(np.int64))

        assert len(self.chosen_samples) == len(self.rejected_samples), "Chosen 和 Rejected 样本数量不匹配！"
        logging.info(f"PreferenceDataset: 成功加载了 {len(self.chosen_samples)} 个偏好对。")

    def __len__(self):
        return len(self.chosen_samples)

    def __getitem__(self, idx):
        chosen_tokens = self.chosen_samples[idx]
        rejected_tokens = self.rejected_samples[idx]

        # [核心新增] 创建 attention mask
        # attention_mask 中，1 表示真实token，0 表示padding
        chosen_mask = (chosen_tokens != self.pad_token_id).long()
        rejected_mask = (rejected_tokens != self.pad_token_id).long()

        return chosen_tokens, rejected_tokens, chosen_mask, rejected_mask


def get_preference_loaders(
        data_dir: Path,
        tokenizer_name: str, # [新增] 需要分词器来获取 pad_token_id
        block_size: int,
        batch_size: int,
        num_workers: int = 0
) -> DataLoader:
    """
    为DPO或RM训练创建数据加载器。
    """
    chosen_file = data_dir / "preference_chosen.bin"
    rejected_file = data_dir / "preference_rejected.bin"

    if not chosen_file.exists() or not rejected_file.exists():
        raise FileNotFoundError(f"在 '{data_dir}' 中未找到 preference_chosen.bin 或 preference_rejected.bin。 "
                                f"请先运行 'data_pipeline/processing/build_preference_bins.py'。")

    tokenizer = Tokenizer.from_file(tokenizer_name)
    pad_token_id = tokenizer.token_to_id("<|pad|>") or tokenizer.token_to_id("<|endoftext|>")

    dataset = PreferenceDataset(chosen_file, rejected_file, block_size, pad_token_id)

    # 简单的返回一个 DataLoader，因为 DPO 和 RM 都需要成对的数据
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# END OF FILE: align/preference_data_loader.py