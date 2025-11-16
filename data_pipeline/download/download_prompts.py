# FILE: data_pipeline/download/download_prompts.py
# -*- coding: utf-8 -*-
"""
[v2.1 - 健壮性修复版] 从 Hugging Face Hub 下载高质量 Prompt 数据集。
- 切换到更稳定、官方维护的 HuggingFaceH4/instruction-dataset 数据集。
- 增加了本地处理步骤，将下载的JSONL转换为简单的txt文件。
"""
from pathlib import Path
import logging
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from datasets import load_dataset
except ImportError:
    logging.error("datasets 未安装。请运行: pip install datasets")
    load_dataset = None


def download_and_process_h4_prompts(limit: int = 5000):
    """
    从 HuggingFaceH4/instruction-dataset 下载数据，并提取 prompts 保存为 txt 文件。
    """
    if not load_dataset:
        return

    repo_id = "HuggingFaceH4/instruction-dataset"
    # [核心修复] 根据错误日志，将数据集分片名称从 "helpful_base" 修正为 "test"
    subset = "test"

    # 将文件保存到 data_pipeline/prompts/ 目录下
    output_dir = Path(__file__).parent.parent / "prompts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "h4_prompts.txt"
    # [同步修改] 更新缓存文件名以保持一致性
    jsonl_cache_path = output_dir / "h4_test.jsonl"

    if output_path.exists():
        logging.info(f"✅ 高质量Prompt数据集已存在于: '{output_path}'")
        return

    logging.info(f"📥 正在从 Hugging Face Hub ({repo_id}) 下载高质量Prompt数据集...")

    try:
        # 使用流式下载以节省内存，并只取需要的部分
        # 使用修正后的 'subset' 变量
        dataset = load_dataset(repo_id, split=f"{subset}[:{limit}]", streaming=False)

        # 将其保存为 jsonl 文件作为缓存，便于调试
        dataset.to_json(jsonl_cache_path)
        logging.info(f"已将原始数据缓存到: '{jsonl_cache_path}'")

        logging.info(f"✍️ 正在处理 JSONL 文件并提取 prompts 到 '{output_path}'...")

        with open(jsonl_cache_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="Extracting prompts"):
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt")
                    if prompt:
                        # 写入文件，每个prompt占一行
                        f_out.write(prompt.strip() + "\n")
                except json.JSONDecodeError:
                    logging.warning(f"跳过无效的JSON行: {line.strip()}")

        logging.info(f"✅ 数据集下载、处理并保存到: '{output_path}'")

    except Exception as e:
        logging.error(f"❌ 下载或处理失败: {e}")
        logging.error("请检查您的网络连接或 datasets 版本。")
        logging.error("如果遇到网络问题，可以尝试设置 HF_ENDPOINT 环境变量。")


if __name__ == "__main__":
    # 我们只下载前5000条作为prompt池，对于CPU训练来说完全足够了
    download_and_process_h4_prompts(limit=5000)
# END OF FILE: data_pipeline/download/download_prompts.py