# FILE: finetune/full/sft_train.py
# -*- coding: utf-8 -*-
"""
[v1.7 - 依赖自动化] SFT (Supervised Fine-Tuning) 训练主脚本
- 在 fast_dev_run 模式下，自动覆盖检查点加载路径。
- [重构] 已迁移到 finetune/full/ 目录。
"""
import torch
import argparse
from pathlib import Path
import time
import sys
import shutil

# --- 路径修复 ---
# 由于文件位置改变，我们需要向上三级才能到达项目根目录
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from finetune.sft_data_loader import get_sft_loaders
from pretrain.components.checkpointing import CheckpointManager
from pretrain.components.training_loop import Trainer

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def main():
    parser = argparse.ArgumentParser(description="[v1.7] 监督微调 (SFT) 脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向SFT配置YAML文件的路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="启用快速开发运行模式，使用固定名称并清理旧目录")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    base_output_dir = Path(cfg.output_dir)
    if args.fast_dev_run:
        run_name = "fast-dev-run"
        output_dir = base_output_dir / "sft" / "full" / run_name
        if output_dir.exists():
            print(f"🧹 fast_dev_run 模式: 正在清理旧的开发目录 {output_dir}")
            shutil.rmtree(output_dir)
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = cfg.run_name.format(timestamp=timestamp)
        output_dir = base_output_dir / "sft" / "full" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)

    # --- 1. 模型 ---
    model = build_model(cfg.model)

    # [核心修改] 自动路径覆盖
    if args.fast_dev_run:
        pretrain_dev_ckpt_path = base_output_dir / "pretrain" / "fast-dev-run" / "checkpoints" / "ckpt_best.pth"
        print(f"🔩 --fast_dev_run: 自动覆盖检查点加载路径。")
        print(f"   - YAML中路径 (将被忽略): {cfg.sft.base_model_checkpoint}")
        print(f"   - 自动解析路径: {pretrain_dev_ckpt_path}")
        cfg.sft.load_from_checkpoint = str(pretrain_dev_ckpt_path)

    if cfg.sft.load_from_checkpoint and Path(cfg.sft.load_from_checkpoint).exists():
        print(f"正在从检查点加载预训练权重: {cfg.sft.load_from_checkpoint}")
        checkpoint = torch.load(cfg.sft.load_from_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 预训练权重加载成功。")
    else:
        print(f"⚠️ 警告：检查点 '{cfg.sft.load_from_checkpoint}' 未找到。将从头开始训练模型。")

    model.to(cfg.device)
    print(f"模型已移动到设备: {cfg.device}")

    # --- 2. 数据 ---
    train_loader, val_loader = get_sft_loaders(
        tokenizer_path=Path(cfg.data.tokenizer_name),
        sft_bin_file=Path(cfg.data.sft_data_path),
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        provide_val_loader=True
    )

    # --- 3. 优化器与调度器 ---
    optimizer = build_optimizer(model, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)
    scaler = GradScaler() if cfg.device == 'cuda' and GradScaler else None

    # --- 4. 检查点 ---
    print("\n--- 4. 初始化检查点管理器 ---")
    sft_ckpt_dir = output_dir / "checkpoints"
    ckpt_manager = CheckpointManager(sft_ckpt_dir, model, optimizer, scheduler, scaler)

    # --- 5. 训练器 ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.device,
        logger=logger,
        ckpt_manager=ckpt_manager,
        hooks=None,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_interval=cfg.logging.log_interval,
        save_interval=len(train_loader),
        scaler=scaler,
        clip_grad_norm=cfg.training.clip_grad_norm,
        loss_spike_threshold=getattr(cfg.training, 'loss_spike_threshold', 5.0),
        max_consecutive_spikes=getattr(cfg.training, 'max_consecutive_spikes', 5),
        grad_norm_history_size=getattr(cfg.training, 'grad_norm_history_size', 100),
        grad_clip_percentile=getattr(cfg.training, 'grad_clip_percentile', 0.9),
        dynamic_clip_factor=getattr(cfg.training, 'dynamic_clip_factor', 1.5)
    )

    trainer.run(cfg.training.max_epochs, 0)


if __name__ == "__main__":
    main()
# END OF FILE: finetune/full/sft_train.py