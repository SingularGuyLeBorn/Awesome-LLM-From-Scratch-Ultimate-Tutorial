# FILE: inference/chat.py
# -*- coding: utf-8 -*-
"""
[v1.7 - 集成量化] 交互式命令行聊天脚本。
- 新增 `--quantize` 参数，支持 Int8 动态量化加速 CPU 推理。
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
from inference.generate import generate_stream
from inference.quantization import Quantizer
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="与你训练的模型进行交互式聊天。")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth) 的路径。")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml) 的路径，用于构建模型结构。")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成时的温度参数。")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成时的top-p (nucleus) 采样参数。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="一次生成的最大token数。")
    parser.add_argument("--quantize", action="store_true", help="[New] 是否对模型进行 Int8 动态量化以加速 CPU 推理。")
    args = parser.parse_args()

    # --- 1. 加载模型和分词器 ---
    print("🚀 正在加载模型和分词器...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    model = build_model(cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer_path = cfg.data.tokenizer_name

    model.eval()
    device = 'cpu' # 动态量化目前主要在 CPU 上有效
    model.to(device)

    # 量化处理逻辑
    if args.quantize:
        print("\n⚖️ 正在应用 Int8 动态量化 (Dynamic Quantization)...")
        print("   这会显著降低内存占用并加速 CPU 推理，但可能会带来微小的精度损失。")
        model = Quantizer.quantize_dynamic(model)
        print("✅ 模型已量化。")
    else:
        try:
            model = model.to(torch.bfloat16)
            print("   -> 模型已转换为 bfloat16 以加速推理。")
        except Exception:
            print("   -> CPU 不支持 bfloat16，将使用 float32。")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    print("✅ 模型加载完成！")
    print("\n--- 开始聊天 (输入 '/quit' 退出, '/clear' 清空历史) ---")

    history = []

    while True:
        try:
            prompt_text = input("😀 > ")
            if prompt_text.lower() == '/quit':
                break
            if prompt_text.lower() == '/clear':
                history = []
                print("\n--- 历史已清空 ---")
                continue

            # --- 2. 格式化输入 ---
            full_prompt_text = ""
            for q, a in history:
                full_prompt_text += f"<|im_start|>{q}<|im_end|>{a}<|endoftext|>"
            full_prompt_text += f"<|im_start|>{prompt_text}<|im_end|>"

            prompt_tokens = tokenizer.encode(full_prompt_text).ids
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

            # --- 3. 流式生成 ---
            print("🤖 > ", end="", flush=True)
            response_tokens = []
            start_time = time.perf_counter()

            token_stream = generate_stream(
                model,
                prompt_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_id=eos_id
            )

            generated_text = ""
            for token_id in token_stream:
                if token_id in [im_end_id, im_start_id, eos_id]:
                    break

                response_tokens.append(token_id)
                new_text = tokenizer.decode(response_tokens)

                newly_generated_part = new_text[len(generated_text):]

                # 净化输出以获得干净的单行打字机效果
                sanitized_part = newly_generated_part.replace('\n', ' ').replace('\r', '')

                print(sanitized_part, end="", flush=True)

                generated_text = new_text

            # --- 4. 结束与统计 ---
            end_time = time.perf_counter()
            duration = end_time - start_time
            num_tokens = len(response_tokens)
            tokens_per_sec = num_tokens / duration if duration > 0 else float('inf')

            final_response = generated_text.replace('\n', ' ').replace('\r', ' ').strip()

            print()
            print(f"   (生成 {num_tokens} tokens, 耗时 {duration:.2f}s, 速度: {tokens_per_sec:.2f} tok/s)")

            history.append((prompt_text, final_response))

        except KeyboardInterrupt:
            print("\n👋 告辞！")
            break
        except Exception as e:
            print(f"\n❌ 出现错误: {e}")


if __name__ == "__main__":
    main()
# END OF FILE: inference/chat.py