# FILE: evaluation/run_leaderboard.py
# -*- coding: utf-8 -*-
"""
[Unified Benchmark Runner] 统一评测入口。
一次运行，评估 MMLU (知识) 和 GSM8K (推理)，生成综合报告。
"""
import argparse
import torch
import json
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model
from tokenizers import Tokenizer
from evaluation.harness.tasks.mmlu import MMLUBenchmark
from evaluation.harness.tasks.gsm8k import GSM8KBenchmark


def main():
    parser = argparse.ArgumentParser(description="LLM Leaderboard Evaluation")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="mmlu,gsm8k", help="Comma separated tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task for fast test")
    args = parser.parse_args()

    # 1. Load Model
    print(f"🚀 Loading Model from {args.checkpoint_path}...")
    cfg = load_config(args.config_path, Path(project_root))
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_name)

    # 某些 tokenizer 没有 padding token，手动指定
    if tokenizer.token_to_id("<|pad|>") is None:
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|endoftext|>"), pad_token="<|endoftext|>")

    model = build_model(cfg.model)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. Run Tasks
    task_list = args.tasks.split(",")
    results = {}

    if "mmlu" in task_list:
        mmlu = MMLUBenchmark()
        results.update(mmlu.evaluate(model, tokenizer, limit=args.limit))

    if "gsm8k" in task_list:
        gsm = GSM8KBenchmark()
        results.update(gsm.evaluate(model, tokenizer, limit=args.limit))

    # 3. Summary
    print("\n" + "=" * 40)
    print("🏆 Leaderboard Score Report")
    print("=" * 40)
    for k, v in results.items():
        print(f"{k:<20}: {v:.2%}")

    # Save
    save_path = Path(project_root) / "evaluation" / "leaderboard_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to {save_path}")


if __name__ == "__main__":
    main()
# END OF FILE: evaluation/run_leaderboard.py