# FILE: evaluation/harness/tasks/gsm8k.py
# -*- coding: utf-8 -*-
"""
[Harness Task] GSM8K Benchmark 实现。
用于 run_leaderboard.py 的统一调用。
"""
import re
from tqdm import tqdm
from datasets import load_dataset
from ..base import Benchmark
from inference.engine.engine import InferenceEngine


class GSM8KBenchmark(Benchmark):
    def __init__(self, shot_num: int = 0):
        super().__init__("gsm8k", shot_num)

    def load_data(self):
        # 使用 'main' 配置加载测试集
        return load_dataset(self.dataset_name, "main", split="test")

    def make_prompt(self, sample) -> str:
        # 标准 CoT Prompt 模板
        question = sample['question']
        return f"<|im_start|>user\nQuestion: {question}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    def extract_answer(self, completion: str) -> str:
        # GSM8K 标准答案通常在 #### 之后
        if "####" in completion:
            return completion.split("####")[1].strip().replace(",", "")
        # 如果没有 ####，尝试提取最后一个数字
        numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
        return numbers[-1] if numbers else ""

    def evaluate(self, model, tokenizer, limit: int = None) -> dict:
        print(f"📚 Evaluating GSM8K (Chain-of-Thought)...")
        ds = self.load_data()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        # 复用统一的推理引擎 (Non-Paged)
        engine = InferenceEngine(model, tokenizer)
        correct_count = 0

        for sample in tqdm(ds, desc="GSM8K"):
            prompt = self.make_prompt(sample)
            # Ground Truth 处理
            ground_truth = self.extract_answer(sample['answer'])

            # 生成 (Greedy Decoding for Math)
            try:
                # generate 返回 list[str]，取第一个
                full_out = engine.generate([prompt], max_new_tokens=256, temperature=0.0)[0]

                # 截取生成的回答部分
                # 注意：engine.generate 返回的是包含 prompt 的完整文本
                # 我们需要根据 prompt 长度截取，或者更智能地解析
                # 简单起见，假设 prompt 是前缀
                if full_out.startswith(prompt):
                    completion = full_out[len(prompt):]
                else:
                    completion = full_out  # Fallback

                model_ans = self.extract_answer(completion)

                # 数值比较
                if model_ans and ground_truth:
                    if abs(float(model_ans) - float(ground_truth)) < 1e-6:
                        correct_count += 1
            except Exception as e:
                # print(f"Error in sample: {e}")
                continue

        acc = correct_count / len(ds)
        print(f"✅ GSM8K Accuracy: {acc:.2%}")
        return {"gsm8k_accuracy": acc}

# END OF FILE: evaluation/harness/tasks/gsm8k.py