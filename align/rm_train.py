# FILE: align/rm_train.py
# -*- coding: utf-8 -*-
"""
[v1.3 - 健壮性修复版] 奖励模型 (Reward Model, RM) 训练主脚本。
- 集成 attention_mask，以实现更精确的奖励值提取。
"""
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import time
import sys

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_reward_model, build_optimizer, build_scheduler, build_loggers
from align.preference_data_loader import get_preference_loaders
from pretrain.components.checkpointing import CheckpointManager
from tqdm import tqdm


def rm_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def main():
    parser = argparse.ArgumentParser(description="奖励模型 (RM) 训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向RM配置YAML文件的路径")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    cfg = load_config(args.config_path, Path(__file__).parent.parent.resolve())

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = cfg.run_name.format(timestamp=timestamp)
    base_output_dir = Path(cfg.output_dir)
    output_dir = base_output_dir / "rlhf" / "rm" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, "rm_run")

    # --- 1. 初始化模型 ---
    reward_model = build_reward_model(cfg.model).to(cfg.device)

    if cfg.rm.load_from_checkpoint:
        print(f"正在从SFT检查点加载权重: {cfg.rm.load_from_checkpoint}")
        checkpoint = torch.load(cfg.rm.load_from_checkpoint, map_location=cfg.device)
        reward_model.transformer.load_state_dict(checkpoint['model_state_dict'])
        print("✅ RM Transformer 权重加载成功。")

    # --- 2. 数据 ---
    train_loader = get_preference_loaders(
        data_dir=Path(cfg.data.data_dir),
        tokenizer_name=cfg.data.tokenizer_name,
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size
    )

    # --- 3. 优化器与调度器 ---
    optimizer = build_optimizer(reward_model, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)

    # --- 4. 检查点管理器 ---
    print("\n--- 4. 初始化检查点管理器 ---")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_manager = CheckpointManager(ckpt_dir, reward_model, optimizer, scheduler)

    # --- 5. 训练循环 ---
    print("\n--- 开始奖励模型 (RM) 训练 ---")
    global_step = 0
    best_accuracy = 0.0
    for epoch in range(cfg.training.max_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [RM Training]")
        total_accuracy = 0
        # [核心修改] 解包 attention_mask
        for chosen_tokens, rejected_tokens, chosen_mask, rejected_mask in pbar:
            chosen_tokens, rejected_tokens = chosen_tokens.to(cfg.device), rejected_tokens.to(cfg.device)
            chosen_mask, rejected_mask = chosen_mask.to(cfg.device), rejected_mask.to(cfg.device)

            # [核心修改] 将 attention_mask 传递给模型
            chosen_rewards = reward_model(chosen_tokens, attention_mask=chosen_mask)
            rejected_rewards = reward_model(rejected_tokens, attention_mask=rejected_mask)

            loss = rm_loss(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            total_accuracy += accuracy

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.2f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % cfg.logging.log_interval == 0:
                logger.log({'rm/loss': loss.item(), 'rm/accuracy_step': accuracy, 'lr': scheduler.get_last_lr()[0]},
                           step=global_step)

            global_step += 1

        epoch_accuracy = total_accuracy / len(train_loader)
        logger.log({'rm/accuracy_epoch': epoch_accuracy}, step=epoch)

        is_best = epoch_accuracy > best_accuracy
        if is_best:
            best_accuracy = epoch_accuracy

        ckpt_manager.save(epoch, 1 - epoch_accuracy, is_best)

    logger.finish()
    print("\n--- 奖励模型训练完成 ---")


if __name__ == "__main__":
    main()
# END OF FILE: align/rm_train.py