# FILE: inference/paged_engine_demo.py
# -*- coding: utf-8 -*-
"""
[新增] PagedAttention 推理引擎的演示脚本。
"""
import torch
import argparse
from pathlib import Path
import sys
import time

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model
from inference.engine.paged_engine import PagedInferenceEngine
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="演示 PagedAttention 推理引擎。")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth) 的路径。")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml) 的路径。")
    args = parser.parse_args()

    # --- 1. 加载模型和分词器 ---
    print("🚀 正在加载模型和分词器...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    model = build_model(cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer_path = cfg.data.tokenizer_name

    device = 'cpu'
    model.to(device)
    try:
        model = model.to(torch.bfloat16)
        print("   -> 模型已转换为 bfloat16 以加速推理。")
    except Exception:
        print("   -> CPU 不支持 bfloat16，将使用 float32。")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("✅ 模型加载完成！")

    # --- 2. 初始化 PagedInferenceEngine ---
    # 假设我们有 256 个物理块，每个块大小为 16 tokens
    engine = PagedInferenceEngine(model, tokenizer, block_size=16, num_blocks=256)
    print("\n🔥 PagedInferenceEngine 初始化完成！")
    print(f"   - 物理块总数: {engine.block_manager.num_blocks}")
    print(f"   - 每块大小: {engine.block_size} tokens")

    # --- 3. 添加推理请求 ---
    prompts = [
        "你好，请介绍一下你自己。",
        "什么是大型语言模型？",
        "请写一首关于宇宙的短诗。",
        "从前有座山，",
    ]
    for i, prompt in enumerate(prompts):
        engine.add_request(prompt, seq_id=i)

    print(f"\n📥 已添加 {len(prompts)} 个推理请求。")

    # --- 4. 运行推理循环 ---
    print("\n⏳ 开始执行推理循环 (step-by-step)...")
    start_time = time.perf_counter()
    step = 0
    all_outputs = {}

    while engine.has_unfinished_requests():
        step_start_time = time.perf_counter()

        finished_this_step = engine.step()

        step_end_time = time.perf_counter()

        # 打印当前步的状态
        num_running = len(engine.scheduler.running)
        num_waiting = len(engine.scheduler.waiting)
        num_finished = len(engine.scheduler.finished)
        free_blocks = engine.block_manager.get_num_free_blocks()

        print(
            f"Step {step:>3}: "
            f"Running: {num_running}, Waiting: {num_waiting}, Finished: {num_finished}, "
            f"Free Blocks: {free_blocks:>3}, "
            f"Time: {(step_end_time - step_start_time) * 1000:.2f} ms"
        )

        if finished_this_step:
            all_outputs.update(finished_this_step)
            for seq_id, text in finished_this_step.items():
                print(f"  ✨ 序列 {seq_id} 已完成！")

        step += 1

    end_time = time.perf_counter()
    print(f"\n✅ 所有请求处理完毕！总耗时: {end_time - start_time:.2f} s")

    # --- 5. 打印最终结果 ---
    print("\n" + "=" * 20 + " 推理结果 " + "=" * 20)
    for i in range(len(prompts)):
        print(f"\n--- Prompt {i} ---\n{prompts[i]}")
        print(f"\n--- Completion {i} ---\n{all_outputs.get(i, 'Error: No output generated')}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
# END OF FILE: inference/paged_engine_demo.py