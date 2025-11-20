# FILE: evaluation/harness/tasks/gsm8k.py
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from ..base import Benchmark
from inference.engine.engine import InferenceEngine  # 复用推理引擎


class GSM8KBenchmark(Benchmark):
    def __init__(self, shot_num: int = 0):
        super().__init__("gsm8k", shot_num)

    def load_data(self):
        return load_dataset(self.dataset_name, "main", split="test")

    def make_prompt(self, sample) -> str:
        # GSM8K Prompt 模板
        question = sample['question']
        return f"<|im_start|>user\nQuestion: {question}\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    def extract_answer(self, completion: str) -> str:
        if "####" in completion:
            return completion.split("####")[1].strip().replace(",", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
        return numbers[-1] if numbers else ""

    def evaluate(self, model, tokenizer, limit: int = None) -> dict:
        print(f"📚 Evaluating GSM8K (Chain-of-Thought)...")
        ds = self.load_data()
        if limit: ds = ds.select(range(min(limit, len(ds))))

        engine = InferenceEngine(model, tokenizer)
        correct_count = 0

        for sample in tqdm(ds, desc="GSM8K"):
            prompt = self.make_prompt(sample)
            ground_truth = self.extract_answer(sample['answer'])

            # 生成 (Greedy)
            try:
                full_out = engine.generate([prompt], max_new_tokens=256, temperature=0.0)[0]
                completion = full_out[len(prompt):]
                model_ans = self.extract_answer(completion)

                if model_ans and float(model_ans) == float(ground_truth):
                    correct_count += 1
            except:
                continue

        acc = correct_count / len(ds)
        print(f"✅ GSM8K Accuracy: {acc:.2%}")
        return {"gsm8k_accuracy": acc}
# END OF FILE: evaluation/harness/tasks/gsm8k.py