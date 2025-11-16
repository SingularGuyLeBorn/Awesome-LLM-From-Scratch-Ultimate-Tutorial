# FILE: data_pipeline/processing/build_midtrain_bins.py
# -*- coding: utf-8 -*-
"""
[新增] 中期训练 (Mid-training) 数据处理脚本
功能：创建一份模拟的、小而精的“高价值”语料。
"""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dummy_midtrain_data(file_path: Path):
    """创建一个微型的、模拟高价值的Mid-train数据集。"""
    if file_path.exists():
        logging.info(f"'{file_path}' 已存在，跳过创建。")
        return

    # 模拟一些更长、更结构化的文本，例如代码或推理
    data = [
        {
            "text": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n# Example usage:\nprint(factorial(5))"},
        {
            "text": "问题：一个农夫有17只羊，除了9只以外都死了，他还剩下几只？\n答案：他还剩下9只羊。这是一个经典的脑筋急转弯问题，关键在于理解“除了...以外”这个短语。"},
        {
            "text": "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the English alphabet."},
        {
            "text": "import torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(128, 64)\n        self.fc2 = nn.Linear(64, 10)\n\n    def forward(self, x):\n        x = F.relu(self.fc1(x))\n        return self.fc2(x)"}
    ]
    with file_path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"已创建虚拟Mid-train数据文件: '{file_path}'")


def process_midtrain_data(
        jsonl_file: Path,
        tokenizer_path: Path,
        output_bin_file: Path,
):
    if not tokenizer_path.exists():
        logging.error(f"分词器文件不存在: {tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    all_token_ids = []

    with jsonl_file.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing mid-train data"):
            record = json.loads(line)
            text = record['text']
            token_ids = tokenizer.encode(text).ids + [eos_id]
            all_token_ids.extend(token_ids)

    # 保存为二进制文件
    np.array(all_token_ids, dtype=np.uint16).tofile(output_bin_file)
    logging.info(f"✅ Mid-train数据处理完成。文件已保存到 '{output_bin_file}'")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "processed_data"
    dummy_file = data_dir / "dummy_midtrain_data.jsonl"
    create_dummy_midtrain_data(dummy_file)

    process_midtrain_data(
        jsonl_file=dummy_file,
        tokenizer_path=data_dir / "tinystories_project_vs4096.json",
        output_bin_file=data_dir / "midtrain_data.bin",
    )
# END OF FILE: data_pipeline/processing/build_midtrain_bins.py