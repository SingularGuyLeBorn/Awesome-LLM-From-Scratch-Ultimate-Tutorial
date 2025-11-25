# FILE: inference/chat.py
# -*- coding: utf-8 -*-
"""
[v2.0 - Universal Chat Interface] 通用对话终端
支持加载本项目全生命周期的模型产物：
1. Full Weights: Pretrain, Full SFT, RM, DPO, PPO, GRPO, GSPO
2. Adapters: LoRA, QLoRA (自动加载 Base + Adapter)
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
from utils.builders import build_model, load_qlora_model_for_inference
from inference.generate import generate_stream
from inference.quantization import Quantizer
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="与你训练的模型进行交互式聊天。")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml)，用于构建模型骨架。")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="模型权重路径 (Base Model 或 Full Finetuned Model)。")
    parser.add_argument("--adapter_path", type=str, default=None, help="[可选] LoRA/QLoRA 适配器权重路径。")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度 (0.0-1.0)。")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P 采样参数。")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成长度。")
    parser.add_argument("--quantize", action="store_true", help="[仅非QLoRA] 启用 Int8 动态量化以加速 CPU 推理。")
    args = parser.parse_args()

    # --- 0. 环境初始化 ---
    print("\n" + "=" * 60)
    print(f"{'🚀 LLM Chat Terminal':^60}")
    print("=" * 60)

    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)
    device = 'cpu'  # 强制 CPU 推理，确保兼容性

    # --- 1. 模型加载逻辑 ---
    model = None

    if args.adapter_path:
        # [模式 A] Base + Adapter (LoRA/QLoRA)
        print(f"🛠️  Mode: [Adapter Fusion]")
        print(f"    Base Model:    {Path(args.checkpoint_path).name}")
        print(f"    Adapter:       {Path(args.adapter_path).name}")

        model = load_qlora_model_for_inference(
            config=cfg,
            base_ckpt_path=args.checkpoint_path,
            adapter_ckpt_path=args.adapter_path,
            device=device
        )

        if args.quantize:
            print("ℹ️  Info: QLoRA 模式已包含 4-bit 量化，忽略 --quantize 参数。")

    else:
        # [模式 B] Full Weights (Pretrain/SFT/RLHF)
        print(f"📦  Mode: [Full Weights]")
        print(f"    Checkpoint:    {Path(args.checkpoint_path).name}")

        print("    -> Building model architecture...")
        model = build_model(cfg.model)

        print(f"    -> Loading state dictionary...")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        # 兼容保存了完整 checkpoint 的情况
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(device)

        # 动态量化与精度处理
        if args.quantize:
            print("    -> Applying Dynamic Int8 Quantization (CPU)...")
            model = Quantizer.quantize_dynamic(model)
        else:
            try:
                model = model.to(torch.bfloat16)
                print("    -> Converted to bfloat16 for inference.")
            except Exception:
                print("    -> CPU does not support bfloat16, using float32.")

    # --- 2. 加载分词器 ---
    tokenizer_path = cfg.data.tokenizer_name
    print(f"📖  Tokenizer:     {Path(tokenizer_path).name}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 特殊 Token ID
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    if im_start_id is None:
        print("⚠️  Warning: Chat tokens not found. Standard completion mode.")

    print("=" * 60)
    print("💡 Tips: 输入 '/quit' 退出, '/clear' 清空历史")
    print("-" * 60)

    # --- 3. 交互循环 ---
    history = []

    while True:
        try:
            prompt_text = input("\n😀 User > ")
            if prompt_text.strip().lower() == '/quit':
                print("👋 Bye!")
                break
            if prompt_text.strip().lower() == '/clear':
                history = []
                print("🧹 History cleared.")
                continue
            if not prompt_text.strip():
                continue

            # 构建 Chat Prompt
            # 格式: <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
            full_prompt_text = ""
            for q, a in history:
                if im_start_id is not None:
                    full_prompt_text += f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
                else:
                    full_prompt_text += f"{q}\n{a}\n"

            if im_start_id is not None:
                full_prompt_text += f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_prompt_text += f"{prompt_text}"

            # 编码
            encoded = tokenizer.encode(full_prompt_text)
            prompt_tokens = torch.tensor([encoded.ids], dtype=torch.long, device=device)

            # 流式生成
            print("🤖 AI   > ", end="", flush=True)
            response_tokens = []
            start_time = time.perf_counter()

            # 确保模型在 eval 模式
            model.eval()

            token_stream = generate_stream(
                model,
                prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_id=eos_id
            )

            generated_text = ""
            for token_id in token_stream:
                # 遇到特殊 token 停止
                if token_id in [im_end_id, eos_id]:
                    break
                # 如果是 im_start，通常不应该生成出来，但也作为停止符处理
                if im_start_id is not None and token_id == im_start_id:
                    break

                response_tokens.append(token_id)
                new_text = tokenizer.decode(response_tokens)

                # 增量打印
                newly_generated_part = new_text[len(generated_text):]

                # 简单的流式输出清洗
                print(newly_generated_part, end="", flush=True)
                generated_text = new_text

            # 统计
            end_time = time.perf_counter()
            duration = end_time - start_time
            num_tokens = len(response_tokens)
            tps = num_tokens / duration if duration > 0 else 0

            print(f"\n\n[Speed: {tps:.2f} tok/s | Time: {duration:.2f}s]")

            history.append((prompt_text, generated_text.strip()))

        except KeyboardInterrupt:
            print("\n⛔ Interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
# END OF FILE: inference/chat.py