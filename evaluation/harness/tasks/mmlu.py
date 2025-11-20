# FILE: evaluation/harness/tasks/mmlu.py
import torch
from tqdm import tqdm
from datasets import load_dataset
from ..base import Benchmark
import numpy as np


class MMLUBenchmark(Benchmark):
    def __init__(self, shot_num: int = 0):
        super().__init__("cais/mmlu", shot_num)
        self.subsets = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
                        "college_computer_science"]
        # 为了演示，只选了几个子集，实际 MMLU 有 57 个

    def load_data(self):
        # MMLU 需要按 subset 加载，这里我们简化逻辑，动态加载
        pass

    def make_prompt(self, sample) -> str:
        # MMLU Prompt 格式:
        # Question: ...
        # A. ...
        # B. ...
        # C. ...
        # D. ...
        # Answer:
        question = sample['question']
        options = sample['choices']
        prompt = f"Question: {question}\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65 + i)}. {opt}\n"
        prompt += "Answer:"
        return prompt

    def evaluate(self, model, tokenizer, limit: int = None) -> dict:
        print(f"📚 Evaluating MMLU ({len(self.subsets)} subsets)...")
        model.eval()
        total_correct = 0
        total_samples = 0

        # 目标 Token 的 ID (A, B, C, D)
        # 注意：分词器不同，ID 可能不同，且可能带空格前缀。这里做简化处理。
        target_tokens = [
            tokenizer.encode("A").ids[-1],
            tokenizer.encode("B").ids[-1],
            tokenizer.encode("C").ids[-1],
            tokenizer.encode("D").ids[-1]
        ]

        for subset in self.subsets:
            try:
                # 加载 test split
                ds = load_dataset(self.dataset_name, subset, split="test")
                if limit: ds = ds.select(range(min(limit, len(ds))))
            except Exception as e:
                print(f"⚠️ Skipping subset {subset}: {e}")
                continue

            subset_correct = 0

            for sample in tqdm(ds, desc=f"MMLU-{subset}", leave=False):
                prompt = self.make_prompt(sample)
                inputs = tokenizer.encode(prompt)
                input_tensor = torch.tensor([inputs.ids], device=model.device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    # 取最后一个 token 的 logits
                    next_token_logits = logits[0, -1, :]

                    # 只比较 A, B, C, D 的概率
                    target_logits = next_token_logits[target_tokens]
                    pred_idx = torch.argmax(target_logits).item()
                    pred_char = chr(65 + pred_idx)

                    if list("ABCD").index(pred_char) == sample['answer']:
                        subset_correct += 1

            total_correct += subset_correct
            total_samples += len(ds)
            print(f"   > {subset}: {subset_correct}/{len(ds)} ({subset_correct / len(ds):.2%})")

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"✅ MMLU Overall Accuracy: {acc:.2%}")
        return {"mmlu_accuracy": acc}
# END OF FILE: evaluation/harness/tasks/mmlu.py