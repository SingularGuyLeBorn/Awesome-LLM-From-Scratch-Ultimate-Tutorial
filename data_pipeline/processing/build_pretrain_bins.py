# FILE: build_pretrain_bins.py
"""
【步骤二：构建二进制文件 - 终极加速版】
真正的零拷贝实现：直接操作 PyArrow 的底层 Buffer
"""
from pathlib import Path
from datasets import load_from_disk
import numpy as np
import logging
import pyarrow as pa
import pyarrow.compute as pc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_bins():
    SPLIT_RATIO = 0.9

    processed_data_path = Path(__file__).parent.parent / "processed_data"
    encoded_dataset_path = processed_data_path / "TinyStories_encoded"
    train_file = processed_data_path / "train.bin"
    val_file = processed_data_path / "val.bin"

    if not encoded_dataset_path.exists():
        logging.error(f"已编码的数据集目录 '{encoded_dataset_path}' 不存在。")
        logging.error("请先运行 'encode_stories.py'。")
        return
    if train_file.exists() and val_file.exists():
        logging.info("✅ 最终的 .bin 文件已存在。跳过构建步骤。")
        return

    logging.info(f"💿 正在从 '{encoded_dataset_path}' 加载已编码的数据集...")
    encoded_dataset = load_from_disk(str(encoded_dataset_path))

    logging.info("🚀 开始终极加速拼接（真零拷贝）...")

    # 方案：直接从每个 chunk 提取扁平化数据，然后用 concatenate
    chunked_array = encoded_dataset.data.column('ids')
    logging.info(f"   PyArrow ChunkedArray 包含 {chunked_array.num_chunks} 个数据块")

    # 收集所有扁平化的 chunk
    flat_chunks = []
    total_tokens = 0

    for i, chunk in enumerate(chunked_array.chunks):
        # chunk 是一个 ListArray，我们要它的 .values (扁平的 Array)
        if hasattr(chunk, 'values'):
            flat_chunk = chunk.values
        elif hasattr(chunk, 'flatten'):
            # 如果是 ListArray，flatten 会给我们扁平数据
            flat_chunk = chunk.flatten()
            if isinstance(flat_chunk, list):
                # 还是 list？转换为 PyArrow Array
                flat_chunk = pa.array(flat_chunk, type=pa.uint16())
        else:
            # 最后的兜底方案
            logging.warning(f"   警告：chunk {i} 类型异常，使用慢速转换")
            flat_chunk = pa.array([item for sublist in chunk.to_pylist() for item in sublist],
                                  type=pa.uint16())

        flat_chunks.append(flat_chunk)
        total_tokens += len(flat_chunk)

        if (i + 1) % 500 == 0:
            logging.info(f"   已处理 {i + 1}/{chunked_array.num_chunks} 个数据块，累计 {total_tokens:,} tokens")

    logging.info(f"   所有数据块处理完成，总计 {total_tokens:,} tokens")

    # 使用 PyArrow 的 concatenate - 这是真正的零拷贝操作
    logging.info("   正在执行 PyArrow concatenate（零拷贝）...")
    all_tokens_arrow = pa.concat_arrays(flat_chunks)

    # 转为 NumPy - 尽可能零拷贝
    logging.info("   正在转换为 NumPy 数组...")
    all_tokens = all_tokens_arrow.to_numpy(zero_copy_only=False)

    # 验证数据类型
    if all_tokens.dtype != np.uint16:
        logging.warning(f"   类型不匹配（{all_tokens.dtype}），执行转换...")
        all_tokens = all_tokens.astype(np.uint16)
    else:
        logging.info(f"   ✅ 数据已为 np.uint16，无需转换")

    logging.info(f"   拼接完成！总 tokens 数: {len(all_tokens):,}")

    # 划分训练集和验证集
    split_index = int(len(all_tokens) * SPLIT_RATIO)
    train_tokens = all_tokens[:split_index]
    val_tokens = all_tokens[split_index:]

    logging.info(f"\n📊 数据集划分:")
    logging.info(f"   - 训练集: {len(train_tokens):,} tokens")
    logging.info(f"   - 验证集: {len(val_tokens):,} tokens")

    # 保存
    logging.info(f"\n💾 保存训练集到 '{train_file}'...")
    train_tokens.tofile(train_file)
    logging.info(f"💾 保存验证集到 '{val_file}'...")
    val_tokens.tofile(val_file)

    logging.info("\n🎉 步骤二完成！可以开始训练了。")


if __name__ == "__main__":
    build_bins()