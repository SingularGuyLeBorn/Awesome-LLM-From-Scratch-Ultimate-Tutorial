# FILE: data_pipeline/processing/process_tinystories.py
"""
【工业级性能版】处理下载好的TinyStories数据集:
采用两遍扫描法，实现内存高效、线性的Token拼接。
"""
import os
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from tokenizers import Tokenizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_tinystories():
    # --- 配置 ---
    TOKENIZER_NAME = "tinystories_project_vs4096.json"

    # 定义数据路径
    raw_data_path = Path(__file__).parent.parent / "downloaded_data" / "TinyStories"
    processed_data_path = Path(__file__).parent.parent / "processed_data"
    tokenizer_path = processed_data_path / TOKENIZER_NAME

    # 检查
    if not raw_data_path.exists():
        logging.error(f"原始数据目录 '{raw_data_path}' 不存在。")
        logging.error("请先运行 'data_pipeline/download/download_tinystories.py' 下载数据。")
        return
    if not tokenizer_path.exists():
        logging.error(f"分词器 '{tokenizer_path}' 不存在。")
        logging.error("请先运行 'data_pipeline/tokenizer/train_tokenizer.py' 训练分词器。")
        return

    processed_data_path.mkdir(parents=True, exist_ok=True)
    train_file = processed_data_path / "train.bin"
    val_file = processed_data_path / "val.bin"

    if train_file.exists() and val_file.exists():
        logging.info("✅ 预处理后的 .bin 文件已存在。跳过处理。")
        return

    # 1. 加载数据集和分词器
    logging.info("💿 正在从本地磁盘加载数据集...")
    dataset = load_from_disk(str(raw_data_path))

    logging.info(f"🤖 正在加载分词器: {TOKENIZER_NAME}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        logging.error("❌ 错误: 分词器中未找到 <|endoftext|> token。")
        return

    # 2. 定义编码函数
    def encode_batch(batch):
        texts = batch['text']
        encodings = tokenizer.encode_batch(texts)
        all_ids = [encoding.ids + [eos_token_id] for encoding in encodings]
        return {"ids": all_ids}

    # 3. 使用.map()进行多进程并行编码
    logging.info("📖 正在使用多进程并行编码所有故事文本...")
    num_proc = max(1, os.cpu_count() // 2)
    logging.info(f"   使用 {num_proc} 个进程进行处理。")

    encoded_dataset = dataset.map(
        encode_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    # --- 核心修复：工业级拼接方案 ---
    logging.info("✨ 编码完成，开始高效拼接...")

    # 4. 第一遍扫描：计算总Token数
    total_tokens_count = sum(len(ids) for ids in tqdm(encoded_dataset['ids'], desc="Pass 1: 计算总长度"))
    logging.info(f"   计算得出总tokens数: {total_tokens_count:,}")

    # 5. 第二遍扫描：一次性分配内存并填充
    all_tokens = np.empty(total_tokens_count, dtype=np.uint16)
    current_position = 0
    for ids in tqdm(encoded_dataset['ids'], desc="Pass 2: 填充数据"):
        length = len(ids)
        all_tokens[current_position: current_position + length] = ids
        current_position += length

    assert current_position == total_tokens_count, "拼接后长度不一致，出现错误！"

    # 6. 划分训练集和验证集
    split_ratio = 0.9
    split_index = int(len(all_tokens) * split_ratio)

    train_tokens = all_tokens[:split_index]
    val_tokens = all_tokens[split_index:]

    logging.info(f"\n📊 数据集划分完毕:")
    logging.info(f"   - 训练集tokens数: {len(train_tokens):,}")
    logging.info(f"   - 验证集tokens数: {len(val_tokens):,}")

    # 7. 保存为二进制文件
    logging.info(f"\n💾 正在将训练集保存到 '{train_file}'...")
    train_tokens.tofile(train_file)
    logging.info(f"💾 正在将验证集保存到 '{val_file}'...")
    val_tokens.tofile(val_file)

    logging.info("\n✅ 数据预处理完成！")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    process_tinystories()
# END OF FILE: data_pipeline/processing/process_tinystories.py