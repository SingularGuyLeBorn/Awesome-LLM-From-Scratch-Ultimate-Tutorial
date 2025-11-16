# FILE: data_pipeline/processing/build_preference_bins.py
"""
[新增] 偏好数据 (Preference Data) 处理脚本
功能：将包含 "chosen" 和 "rejected" 的JSONL格式数据转换为二进制 .bin 文件。
"""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dummy_preference_data(file_path: Path):
    """创建一个微型的偏好数据集用于演示。"""
    if file_path.exists():
        logging.info(f"'{file_path}' 已存在，跳过创建。")
        return

    data = [
        {"chosen": "你好！有什么可以帮助你的吗？", "rejected": "你好。"},
        {"chosen": "大型语言模型（LLM）是一种经过大量文本数据训练的人工智能模型。", "rejected": "不知道。"},
        {"chosen": "天空的蓝色是由于瑞利散射效应。", "rejected": "因为它是蓝色的。"}
    ]
    with file_path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"已创建虚拟偏好数据文件: '{file_path}'")


def process_preference_data(
        jsonl_file: Path,
        tokenizer_path: Path,
        output_dir: Path,
        max_seq_len: int
):
    if not tokenizer_path.exists():
        logging.error(f"分词器文件不存在: {tokenizer_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    chosen_bin_file = output_dir / "preference_chosen.bin"
    rejected_bin_file = output_dir / "preference_rejected.bin"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    all_chosen_ids = []
    all_rejected_ids = []

    with jsonl_file.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing preference data"):
            record = json.loads(line)
            chosen_text = record['chosen']
            rejected_text = record['rejected']

            chosen_ids = tokenizer.encode(chosen_text).ids + [eos_id]
            rejected_ids = tokenizer.encode(rejected_text).ids + [eos_id]

            # 填充或截断到统一长度
            chosen_ids = chosen_ids[:max_seq_len] + [eos_id] * (max_seq_len - len(chosen_ids))
            rejected_ids = rejected_ids[:max_seq_len] + [eos_id] * (max_seq_len - len(rejected_ids))

            all_chosen_ids.extend(chosen_ids)
            all_rejected_ids.extend(rejected_ids)

    # 保存为二进制文件
    np.array(all_chosen_ids, dtype=np.uint16).tofile(chosen_bin_file)
    np.array(all_rejected_ids, dtype=np.uint16).tofile(rejected_bin_file)

    logging.info(f"✅ 偏好数据处理完成。文件已保存到 '{output_dir}'")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "processed_data"
    create_dummy_preference_data(data_dir / "dummy_preference_data.jsonl")

    process_preference_data(
        jsonl_file=data_dir / "dummy_preference_data.jsonl",
        tokenizer_path=data_dir / "tinystories_project_vs4096.json",
        output_dir=data_dir,
        max_seq_len=256
    )
# END OF FILE: data_pipeline/processing/build_preference_bins.py