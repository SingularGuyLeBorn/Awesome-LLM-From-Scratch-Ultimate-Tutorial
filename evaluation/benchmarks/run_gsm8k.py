# FILE: evaluation/benchmarks/run_gsm8k.py
# -*- coding: utf-8 -*-
"""
[新增] GSM8K 数学推理能力评测脚本 (Lightweight版)。

功能:
1. 加载指定的 Hugging Face 数据集 (GSM8K)。
2. 使用我们的 InferenceEngine 进行少样本 (Few-shot) 或零样本 (Zero-shot) 推理。
3. 使用正则表达式提取模型答案中的数字。
4. 与标准答案进行比较并计算准确率。
"""
import argparse
import torch
import re
import sys
from pathlib import Path
from tqdm import tqdm
import json

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model
from inference.engine.engine import InferenceEngine
from tokenizers import Tokenizer

try:
    from datasets import load_dataset
except ImportError:
    print("请先安装 datasets 库: pip install datasets")
    sys.exit(1)


def extract_answer(completion: str) -> str:
    """
    从 GSM8K 的回答中提取最终的数值答案。
    GSM8K 的标准回答通常以 "#### " 结尾，后跟数字。
    如果模型没有遵循此格式，我们尝试提取最后一个数字。
    """
    # 1. 尝试标准的 "#### " 格式
    if "####" in completion:
        answer = completion.split("####")[1].strip()
        # 移除可能存在的逗号，例如 1,234 -> 1234
        return answer.replace(",", "")

    # 2. 备选：提取文本中的最后一个数字
    # 匹配整数或小数
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
    if numbers:
        return numbers[-1]

    return ""


def is_correct(model_answer: str, ground_truth: str) -> bool:
    """比较模型答案和标准答案是否数值相等。"""
    try:
        return float(model_answer) == float(ground_truth)
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description="GSM8K Benchmark 评测")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth)")
    parser.add_argument("--limit", type=int, default=100, help="评测样本数量限制 (为了速度)")
    parser.add_argument("--quantize", action="store_true", help="使用 Int8 量化加速评测")
    parser.add_argument("--shot", type=int, default=0, help="Few-shot 样本数 (0=Zero-shot)")
    args = parser.parse_args()

    # 1. 加载配置和模型
    print("🚀 [GSM8K] 初始化...")
    cfg = load_config(args.config_path, Path(project_root))
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_name)

    # 强制将 pad token 设置为 eos token，以便进行 batch padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|endoftext|>"), pad_token="<|endoftext|>")

    model = build_model(cfg.model)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 初始化推理引擎 (支持 batch)
    engine = InferenceEngine(model, tokenizer, quantize=args.quantize)

    # 2. 加载数据
    print("📚 正在加载 GSM8K 数据集 (split='test')...")
    # 使用 'main' 配置
    dataset = load_dataset("gsm8k", "main", split=f"test[:{args.limit}]")

    print(f"🔥 开始评测 (Samples: {len(dataset)}, Quantize: {args.quantize})...")

    correct_count = 0
    results = []

    # 定义 Prompt 模板
    # 对于 Base 模型，我们使用续写风格；对于 Instruct 模型，建议使用 Chat 模板
    # 这里假设是 Instruct/Chat 模型
    prompt_template = "<|im_start|>user\nQuestion: {question}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    # 3. 逐个评测 (未来可以优化为 Batch 评测)
    pbar = tqdm(dataset)
    for sample in pbar:
        question = sample['question']
        # 提取 Ground Truth (GSM8K 格式: ".... #### 42")
        ground_truth = extract_answer(sample['answer'])

        prompt = prompt_template.format(question=question)

        # 生成
        try:
            # 获取生成的回答部分
            # generate 返回的是 prompt + completion，我们需要截取
            full_output = engine.generate([prompt], max_new_tokens=256, temperature=0.0)[0]  # Greedy decoding
            completion = full_output[len(prompt):]

            model_val = extract_answer(completion)

            if is_correct(model_val, ground_truth):
                correct_count += 1
                res_str = "✅ Correct"
            else:
                res_str = "❌ Wrong"

            pbar.set_postfix(acc=f"{correct_count / (pbar.n + 1):.2%}")

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_val,
                "completion": completion,
                "correct": is_correct(model_val, ground_truth)
            })

        except Exception as e:
            print(f"\nError processing sample: {e}")

    accuracy = correct_count / len(dataset)
    print("\n" + "=" * 50)
    print(f"📊 GSM8K Evaluation Result")
    print(f"Samples: {len(dataset)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 50)

    # 保存结果
    output_file = Path(project_root) / "evaluation" / f"gsm8k_results_{accuracy:.2f}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
# END OF FILE: evaluation/benchmarks/run_gsm8k.py