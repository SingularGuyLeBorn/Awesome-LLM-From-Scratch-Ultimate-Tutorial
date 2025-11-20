# FILE: finetune/distillation/distill_train.py
# -*- coding: utf-8 -*-
"""
[Knowledge Distillation] 知识蒸馏训练脚本。
Teacher: 冻结，只负责推理产生 Soft Targets。
Student: 训练，学习 Hard Labels 和 Soft Targets。
"""
import torch
import argparse
from pathlib import Path
import time
import sys
import shutil

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from pretrain.data_loader import get_pretrain_loaders  # 复用预训练数据加载器
from pretrain.components.checkpointing import CheckpointManager
from finetune.distillation.loss import DistillationLoss
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="[Distillation] Knowledge Distillation Training")
    parser.add_argument("--config_path", type=str, required=True, help="Path to Distillation config YAML")
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast dev run mode")
    args = parser.parse_args()

    # 0. Load Config
    base_path = Path(__file__).parent.parent.parent.resolve()
    cfg = load_config(args.config_path, base_path)

    # Setup Directories
    base_output_dir = Path(cfg.output_dir)
    run_name = "fast-dev-run" if args.fast_dev_run else cfg.run_name.format(timestamp=time.strftime('%Y%m%d-%H%M%S'))
    output_dir = base_output_dir / "distillation" / run_name
    if args.fast_dev_run and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)
    device = cfg.device

    # 1. Build Teacher Model (Frozen)
    print("\n--- 1. Building Teacher Model ---")
    teacher_model = build_model(cfg.teacher_model)

    # Load Teacher Checkpoint
    teacher_ckpt_path = cfg.distillation.teacher_checkpoint
    if args.fast_dev_run:
        # 自动寻找一个存在的检查点作为 Teacher
        teacher_ckpt_path = str(base_output_dir / "pretrain/fast-dev-run/checkpoints/ckpt_best.pth")
        print(f"🔩 Fast Dev: Using pretrain ckpt as Teacher: {teacher_ckpt_path}")

    if Path(teacher_ckpt_path).exists():
        print(f"Loading Teacher weights from: {teacher_ckpt_path}")
        ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
        teacher_model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("⚠️ Warning: Teacher checkpoint not found! Using random weights (Testing only).")

    teacher_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 2. Build Student Model (Trainable)
    print("\n--- 2. Building Student Model ---")
    student_model = build_model(cfg.student_model)
    student_model.to(device)

    # 3. Data, Optimizer
    print("\n--- 3. Setup Data & Optimizer ---")
    train_loader, val_loader = get_pretrain_loaders(
        tokenizer_name=cfg.data.tokenizer_name,
        data_dir=Path(cfg.data.data_dir),
        block_size=cfg.student_model.max_seq_len,  # 使用 Student 的上下文长度
        batch_size=cfg.training.batch_size,
        train_data_limit=getattr(cfg.data, 'train_data_limit', None),
        val_data_limit=getattr(cfg.data, 'val_data_limit', None)
    )

    optimizer = build_optimizer(student_model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training, len(train_loader) * cfg.training.max_epochs)
    criterion = DistillationLoss(temperature=cfg.distillation.temperature, alpha=cfg.distillation.alpha)
    ckpt_manager = CheckpointManager(output_dir / "checkpoints", student_model, optimizer, scheduler)

    # 4. Distillation Loop
    print("\n🚀 Starting Distillation...")
    global_step = 0

    for epoch in range(cfg.training.max_epochs):
        student_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Distill]")

        for x, y, loss_mask in pbar:
            x, y = x.to(device), y.to(device)

            # Teacher Forward (No Grad)
            with torch.no_grad():
                teacher_logits = teacher_model(x)

            # Student Forward
            student_logits = student_model(x)

            # Compute Loss
            loss, ce_loss, kd_loss = criterion(student_logits, teacher_logits, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Clip Grad
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.training.clip_grad_norm)

            optimizer.step()
            scheduler.step()

            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce_loss.item():.4f}", kd=f"{kd_loss.item():.4f}")

            if global_step % cfg.logging.log_interval == 0:
                logger.log({
                    "train/loss": loss.item(),
                    "train/ce_loss": ce_loss.item(),
                    "train/kd_loss": kd_loss.item(),
                    "lr": scheduler.get_last_lr()[0]
                }, step=global_step)

        # Simple Validation (Loss only)
        ckpt_manager.save(epoch, loss.item())

    logger.finish()


if __name__ == "__main__":
    main()
# END OF FILE: finetune/distillation/distill_train.py