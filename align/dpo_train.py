# FILE: align/dpo_train.py
"""
[v1.1 - 目录重构版] DPO (Direct Preference Optimization) 训练主脚本
- 输出目录将自动保存到 runs/rlhf/dpo/ 下。
"""
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import time
import sys
from copy import deepcopy

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from align.preference_data_loader import get_preference_loaders
from pretrain.components.checkpointing import CheckpointManager
from align.algorithms.dpo.implementation import dpo_loss
from tqdm import tqdm


def get_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != -1
    gathered_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered_log_probs * mask


def main():
    parser = argparse.ArgumentParser(description="DPO 对齐训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向DPO配置YAML文件的路径")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    # [核心修改] 自动创建层级化输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = cfg.run_name.format(timestamp=timestamp)
    base_output_dir = Path(cfg.output_dir)
    output_dir = base_output_dir / "rlhf" / "dpo" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)

    # --- 1. 初始化模型 ---
    policy_model = build_model(cfg.model).to(cfg.device)
    reference_model = deepcopy(policy_model).to(cfg.device)
    for param in reference_model.parameters():
        param.requires_grad = False

    print(f"正在从SFT检查点加载权重: {cfg.dpo.load_from_checkpoint}")
    checkpoint = torch.load(cfg.dpo.load_from_checkpoint, map_location=cfg.device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    reference_model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Policy 和 Reference 模型权重加载成功。")

    # --- 2. 数据 ---
    train_loader = get_preference_loaders(
        data_dir=Path(cfg.data.data_dir),
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size
    )

    # --- 3. 优化器与调度器 ---
    optimizer = build_optimizer(policy_model, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)

    # --- 4. 训练循环 ---
    print("\n--- 开始DPO训练 ---")
    global_step = 0
    for epoch in range(cfg.training.max_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [DPO Training]")
        for chosen_tokens, rejected_tokens in pbar:
            chosen_tokens, rejected_tokens = chosen_tokens.to(cfg.device), rejected_tokens.to(cfg.device)

            policy_chosen_logits = policy_model(chosen_tokens)
            policy_rejected_logits = policy_model(rejected_tokens)

            policy_chosen_logps = get_log_probs(policy_chosen_logits, chosen_tokens)
            policy_rejected_logps = get_log_probs(policy_rejected_logits, rejected_tokens)

            with torch.no_grad():
                reference_chosen_logits = reference_model(chosen_tokens)
                reference_rejected_logits = reference_model(rejected_tokens)

                reference_chosen_logps = get_log_probs(reference_chosen_logits, chosen_tokens)
                reference_rejected_logps = get_log_probs(reference_rejected_logits, rejected_tokens)

            loss = dpo_loss(
                policy_chosen_logps.sum(-1), policy_rejected_logps.sum(-1),
                reference_chosen_logps.sum(-1), reference_rejected_logps.sum(-1),
                beta=cfg.dpo.beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % cfg.logging.log_interval == 0:
                logger.log({'dpo/loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}, step=global_step)

            global_step += 1

    logger.finish()
    print("\n--- DPO训练完成 ---")


if __name__ == "__main__":
    main()
# END OF FILE: align/dpo_train.py