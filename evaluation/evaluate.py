# FILE: evaluation/evaluate.py
# -*- coding: utf-8 -*-
"""
[新增] 通用模型评估脚本。
功能:
- 加载指定的模型检查点。
- 在验证集上计算平均损失。
- 计算并报告困惑度 (Perplexity)。
"""
import torch
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import torch.nn.functional as F

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model
from pretrain.data_loader import get_pretrain_loaders
from evaluation.metrics.perplexity import calculate_perplexity


@torch.no_grad()
def run_evaluation(model, val_loader, device):
    """在验证集上运行评估。"""
    model.eval()
    total_loss = 0
    pbar = tqdm(val_loader, desc="[Evaluating]")

    for x, y, loss_mask in pbar:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cpu' else torch.float16,
                            enabled=True):
            logits = model(x)

            # 计算损失
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss_mask_flat = loss_mask.view(-1)

            loss = F.cross_entropy(logits_flat, y_flat, reduction='none', ignore_index=-1)
            masked_loss = loss * loss_mask_flat
            avg_loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-9)

        total_loss += avg_loss.item()
        pbar.set_postfix(loss=f"{avg_loss.item():.4f}")

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml) 的路径。")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth) 的路径。")
    parser.add_argument("--eval_data_limit", type=int, default=1000, help="用于评估的验证集文档数量上限。")
    args = parser.parse_args()

    # --- 1. 加载配置、模型和分词器 ---
    print("🚀 正在加载配置和模型...")
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    model = build_model(cfg.model)

    checkpoint = torch.load(args.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 成功从 '{args.checkpoint_path}' 加载模型权重。")

    model.to(cfg.device)

    # --- 2. 准备数据加载器 ---
    print("\n📚 正在准备验证数据加载器...")
    _, val_loader = get_pretrain_loaders(
        tokenizer_name=cfg.data.tokenizer_name,
        data_dir=Path(cfg.data.data_dir),
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        train_data_limit=0,  # 我们不需要训练集
        val_data_limit=args.eval_data_limit,
        return_train_loader=False  # 只返回验证加载器
    )

    # --- 3. 运行评估 ---
    print("\n🔬 开始评估...")
    avg_loss = run_evaluation(model, val_loader, cfg.device)
    perplexity = calculate_perplexity(avg_loss)

    # --- 4. 报告结果 ---
    print("\n" + "=" * 50)
    print(f"{'评估结果':^50}")
    print("=" * 50)
    print(f"  - 模型检查点: {args.checkpoint_path}")
    print(f"  - 验证集文档数: {args.eval_data_limit}")
    print("-" * 50)
    print(f"  - 平均验证损失 (Avg. Loss): {avg_loss:.4f}")
    print(f"  - 困惑度 (Perplexity): {perplexity:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
# END OF FILE: evaluation/evaluate.py