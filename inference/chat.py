# FILE: inference/chat.py
# -*- coding: utf-8 -*-
"""
[v1.6 - 输出净化] 交互式命令行聊天脚本。
- 增加输出净化逻辑，将模型生成的换行符替换为空格，以保证终端单行打字机效果。
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
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="与你训练的模型进行交互式聊天。")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth) 的路径。")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml) 的路径，用于构建模型结构。")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成时的温度参数。")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成时的top-p (nucleus) 采样参数。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="一次生成的最大token数。")
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
    device = 'cpu'
    model.to(device)
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

                # [核心修复] 净化输出以获得干净的单行打字机效果
                # 模型可能会生成 \n, \r 等字符，直接打印会导致终端显示混乱。
                # 我们将其替换为空格，确保光标始终在同一行。
                sanitized_part = newly_generated_part.replace('\n', ' ').replace('\r', '')

                print(sanitized_part, end="", flush=True)

                generated_text = new_text

            # --- 4. 结束与统计 ---
            end_time = time.perf_counter()
            duration = end_time - start_time
            num_tokens = len(response_tokens)
            tokens_per_sec = num_tokens / duration if duration > 0 else float('inf')

            # 最终的完整回答也需要被净化，以便在历史记录中保持整洁
            final_response = generated_text.replace('\n', ' ').replace('\r', ' ').strip()

            # 由于我们是增量打印，最后需要打印一个换行符来结束这一行
            print()
            print(f"   (生成 {num_tokens} tokens, 耗时 {duration:.2f}s, 速度: {tokens_per_sec:.2f} tok/s)")

            # 更新历史
            history.append((prompt_text, final_response))

        except KeyboardInterrupt:
            print("\n👋 告辞！")
            break
        except Exception as e:
            print(f"\n❌ 出现错误: {e}")


if __name__ == "__main__":
    main()
# END OF FILE: inference/chat.py