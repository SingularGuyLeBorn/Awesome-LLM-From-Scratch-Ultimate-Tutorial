# FILE: align/prompt_loader.py
# -*- coding: utf-8 -*-
"""
[新增] 为在线对齐算法 (PPO, GSPO) 提供真实 Prompt 的数据加载器。
"""
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from pathlib import Path
from typing import List, Dict
import logging


class PromptDataset(Dataset):
    """
    一个简单的数据集，用于从文本文件中加载 prompts。
    """

    def __init__(self, prompt_file_path: Path):
        self.prompt_file_path = prompt_file_path

        if not prompt_file_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_file_path}")

        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            # 读取所有行，并过滤掉空行或注释行
            self.prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        logging.info(f"成功从 '{prompt_file_path.name}' 加载了 {len(self.prompts)} 条 prompts。")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


class DataCollatorForPrompt:
    """
    一个数据整理器，用于对一批文本 prompts 进行分词和左填充。
    """

    def __init__(self, tokenizer: Tokenizer, max_prompt_len: int):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_token_id is None:
            # 如果没有 pad token，可以用 eos token 替代
            self.pad_token_id = tokenizer.token_to_id("<|endoftext|>")
            logging.warning("分词器中未找到 '<|pad|>' token，将使用 '<|endoftext|>' 作为填充符。")

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        # 对批次内的所有 prompts 进行编码
        encoded_prompts = self.tokenizer.encode_batch(batch)

        batch_tokens = []
        for prompt in encoded_prompts:
            tokens = prompt.ids
            # 截断到最大长度
            if len(tokens) > self.max_prompt_len:
                tokens = tokens[:self.max_prompt_len]
            batch_tokens.append(tokens)

        # 找到这个批次中最长的 prompt 长度
        max_len_in_batch = max(len(tokens) for tokens in batch_tokens)

        # 进行左填充
        padded_batch = []
        for tokens in batch_tokens:
            padding_needed = max_len_in_batch - len(tokens)
            padded_tokens = [self.pad_token_id] * padding_needed + tokens
            padded_batch.append(padded_tokens)

        return torch.tensor(padded_batch, dtype=torch.long)


def get_prompt_loader(
        prompt_file_path: Path,
        tokenizer: Tokenizer,
        batch_size: int,
        max_prompt_len: int,
        num_workers: int = 0
) -> DataLoader:
    """
    构建并返回一个用于加载和处理 prompts 的 DataLoader。
    """
    dataset = PromptDataset(prompt_file_path)
    collator = DataCollatorForPrompt(tokenizer, max_prompt_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers
    )

# END OF FILE: align/prompt_loader.py