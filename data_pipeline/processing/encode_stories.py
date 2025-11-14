# FILE: data_pipeline/processing/encode_stories.py
"""
【步骤一：编码 - 性能优化版】
单一职责：加载原始数据集，使用指定分词器进行多进程并行编码，
并将结果以高效的Arrow格式保存为中间文件。

优化点:
1. 增大 batch_size 以提高 map 吞吐量。
2. 调整CPU核心数（例如 os.cpu_count() - 2），可以根据系统调整。
3. (关键) 在保存前，并行地将 token ID 强制转换为 uint16，
   这是为了极大加速下一步骤 (build_pretrain_bins.py)，
   避免其在主进程中进行昂贵的内存复制和类型转换。
"""
import os
from pathlib import Path
from datasets import load_from_disk, Features, Value, Sequence
from tokenizers import Tokenizer
import logging
from multiprocessing import freeze_support

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def encode_stories():
    # --- 配置 ---
    TOKENIZER_NAME = "tinystories_project_vs4096.json"
    # 优化点: 为 map 操作设置一个更大的批处理大小，
    # 尤其是当处理像 TinyStories 这样的短文本时，
    # 这可以显著减少进程调度的开销。
    PROCESSING_BATCH_SIZE = 10_000

    # 定义路径
    raw_data_path = Path(__file__).parent.parent / "downloaded_data" / "TinyStories"
    processed_data_path = Path(__file__).parent.parent / "processed_data"
    tokenizer_path = processed_data_path / TOKENIZER_NAME
    output_path = processed_data_path / "TinyStories_encoded"

    # --- 检查 ---
    if not raw_data_path.exists():
        logging.error(f"原始数据目录 '{raw_data_path}' 不存在。")
        return
    if not tokenizer_path.exists():
        logging.error(f"分词器 '{tokenizer_path}' 不存在。")
        return
    if output_path.exists():
        logging.info(f"✅ 已编码的数据集 '{output_path}' 已存在。跳过编码步骤。")
        return

    # 1. 加载
    logging.info("💿 正在从本地磁盘加载原始数据集...")
    dataset = load_from_disk(str(raw_data_path))

    logging.info(f"🤖 正在加载分词器: {TOKENIZER_NAME}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        logging.error("❌ 错误: 分词器中未找到 <|endoftext|> token。")
        return
    # 假设词汇表大小小于 65535 (uint16 的最大值)
    if eos_token_id > 65535 or tokenizer.get_vocab_size() > 65536:
        logging.warning("⚠️ 警告: 词汇表大小或 EOS token ID 超出 uint16 范围。")
        logging.warning("   如果下一步的 'build_pretrain_bins.py' 失败，")
        logging.warning("   请将此脚本中的 'uint16' 改为 'uint32'。")


    # 2. 定义编码函数
    def encode_batch(batch):
        texts = batch['text']
        encodings = tokenizer.encode_batch(texts)
        # 注意：这里返回的是一个包含 token ID 列表的列表
        all_ids = [encoding.ids + [eos_token_id] for encoding in encodings]
        return {"ids": all_ids}

    # 3. 多进程并行编码
    logging.info("📖 正在使用多进程并行编码所有故事文本...")
    # 优化点: 调整进程数。
    # `os.cpu_count() // 2` 是一个保守的选择。
    # 尝试 `os.cpu_count() - 2` (留出系统和I/O核心) 可能会更快。
    num_proc = max(1, os.cpu_count() - 2 if os.cpu_count() > 2 else 1)
    logging.info(f"   使用 {num_proc} 个进程进行处理 (batch_size={PROCESSING_BATCH_SIZE})。")

    encoded_dataset = dataset.map(
        encode_batch,
        batched=True,
        num_proc=num_proc,
        batch_size=PROCESSING_BATCH_SIZE, # 优化点: 增大 batch_size
        remove_columns=dataset.column_names
    )

    # 4. 关键优化：在保存前转换类型
    # 这是为了优化下一步 (build_pretrain_bins.py)。
    # 在这里使用多进程将 'ids' 列转换为 uint16，
    # 这样下一步脚本加载数据时，数据已经是正确的类型 (np.uint16)，
    # 从而避免在主进程中执行一个巨大、昂贵、且占用大量内存的 .astype() 操作。
    logging.info(f"   ...编码完成。正在并行转换为 uint16...")
    target_features = Features({
        'ids': Sequence(Value('uint16'))
    })
    encoded_dataset = encoded_dataset.cast(target_features, num_proc=num_proc)
    logging.info("   ...类型转换完成。")


    # 5. 保存为Arrow格式
    logging.info(f"💾 正在将编码后的 (uint16) 数据集以Arrow格式保存到: {output_path}")
    encoded_dataset.save_to_disk(str(output_path))

    logging.info("\n✅ 步骤一：编码（优化版）完成！")
    logging.info(f"下一步，请运行 'build_pretrain_bins.py' 来构建最终的 .bin 训练文件。")


if __name__ == "__main__":
    # 在Windows上，为了让多进程正常工作，需要这行代码
    freeze_support()
    encode_stories()
# END OF FILE: data_pipeline/processing/encode_stories.py