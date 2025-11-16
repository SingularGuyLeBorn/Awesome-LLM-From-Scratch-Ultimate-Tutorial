# FILE: data_pipeline/processing/build_sft_bins.py
"""
[v1.1 - 索引优化版] SFT 数据处理脚本
功能：将JSONL格式的指令数据转换为模型可以训练的二进制 .bin 文件。
- 新增：在生成 .bin 文件后，为其创建一个 .idx.npy 索引文件以加速加载。
"""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dummy_sft_data(file_path: Path):
    """创建一个微型的SFT数据集用于演示。"""
    if file_path.exists():
        logging.info(f"'{file_path}' 已存在，跳过创建。")
        return

    data = [
        {"prompt": "你好", "response": "你好！有什么可以帮助你的吗？"},
        {"prompt": "什么是大型语言模型？",
         "response": "大型语言模型（LLM）是一种经过大量文本数据训练的人工智能模型，能够理解和生成类似人类的文本。"},
        {"prompt": "请写一首关于宇宙的短诗。", "response": "银河旋臂舞，星辰作尘埃。\n光年一瞬过，寂静中醒来。"},
        {"prompt": "1 + 1 = ?", "response": "2"},
        {"prompt": "天空为什么是蓝色的？",
         "response": "天空的蓝色是由于瑞利散射效应。当太阳光穿过地球大气层时，空气中的氮气和氧气等小分子会散射蓝光（波长较短）的效率远高于红光（波长较长），所以我们看到了蓝色的天空。"}
    ]
    with file_path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"已创建虚拟SFT数据文件: '{file_path}'")


def process_sft_data(
        jsonl_file: Path,
        tokenizer_path: Path,
        output_bin_file: Path,
        max_seq_len: int
):
    if not jsonl_file.exists():
        logging.error(f"SFT数据文件不存在: {jsonl_file}")
        return
    if not tokenizer_path.exists():
        logging.error(f"分词器文件不存在: {tokenizer_path}")
        return

    logging.info(f"加载分词器: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # 获取特殊token ID
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    if any(id is None for id in [im_start_id, im_end_id, eos_id]):
        raise ValueError("分词器缺少必要的特殊token: <|im_start|>, <|im_end|>, <|endoftext|>")

    all_token_ids = []

    logging.info(f"正在处理SFT数据文件: {jsonl_file}")
    with jsonl_file.open('r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing SFT data"):
            record = json.loads(line)
            prompt = record['prompt']
            response = record['response']

            prompt_ids = tokenizer.encode(prompt).ids
            response_ids = tokenizer.encode(response).ids

            # 格式化: <|im_start|> prompt <|im_end|> response <|endoftext|>
            token_ids = [im_start_id] + prompt_ids + [im_end_id] + response_ids + [eos_id]

            # 截断
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]

            all_token_ids.extend(token_ids)

    logging.info(f"总计 tokens 数量: {len(all_token_ids):,}")

    # 保存为二进制文件
    arr = np.array(all_token_ids, dtype=np.uint16)
    logging.info(f"正在保存到二进制文件: {output_bin_file}")
    arr.tofile(output_bin_file)
    logging.info("✅ SFT数据处理完成。")

    # [核心新增] 创建并保存索引文件
    logging.info("正在为 .bin 文件创建索引以加速加载...")
    # 使用 memmap 重新加载数据以节省内存
    data_for_indexing = np.memmap(output_bin_file, dtype=np.uint16, mode='r')
    boundaries = np.where(data_for_indexing == eos_id)[0]
    index_file = output_bin_file.with_suffix('.idx.npy')
    np.save(index_file, boundaries)
    logging.info(f"✅ 索引文件已创建: {index_file} (包含 {len(boundaries)} 个样本边界)")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "processed_data"
    data_dir.mkdir(exist_ok=True)

    dummy_file = data_dir / "dummy_sft_data.jsonl"
    create_dummy_sft_data(dummy_file)

    tokenizer_file = data_dir / "tinystories_project_vs4096.json"
    output_file = data_dir / "sft_data.bin"

    process_sft_data(
        jsonl_file=dummy_file,
        tokenizer_path=tokenizer_file,
        output_bin_file=output_file,
        max_seq_len=256
    )
# END OF FILE: data_pipeline/processing/build_sft_bins.py