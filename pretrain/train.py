# FILE: pretrain/train.py
# -*- coding: utf-8 -*-
"""
【v3.6 - 开发模式增强】预训练主脚本。
- 新增 --fast_dev_run 命令行标志，用于固定运行名称并自动清理目录，方便快速迭代。
"""
import torch
import argparse
from pathlib import Path
import time
import sys
import shutil

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from pretrain.data_loader import get_pretrain_loaders
from pretrain.components.checkpointing import CheckpointManager
from pretrain.components.training_loop import Trainer
from pretrain.components.hooks import register_hooks

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def main():
    parser = argparse.ArgumentParser(description="[v3.6] 统一预训练/继续预训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向配置YAML文件的路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="启用快速开发运行模式，使用固定名称并清理旧目录")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    # [核心修改] 根据 --fast_dev_run 标志决定运行名称和目录
    base_output_dir = Path(cfg.output_dir)
    if args.fast_dev_run:
        run_name = "fast-dev-run"
        output_dir = base_output_dir / "pretrain" / run_name
        if output_dir.exists():
            print(f"🧹 fast_dev_run 模式: 正在清理旧的开发目录 {output_dir}")
            shutil.rmtree(output_dir)
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = cfg.run_name.format(timestamp=timestamp)
        output_dir = base_output_dir / "pretrain" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)
    print(f"配置加载自: {args.config_path}")
    print(f"所有输出将保存到: {output_dir}")

    # --- 1. 模型 ---
    model = build_model(cfg.model)
    model.to(cfg.device)
    print(f"模型已移动到设备: {cfg.device}")

    # --- 2. 优化器、调度器、混合精度 ---
    optimizer = build_optimizer(model, cfg.training)
    train_limit = getattr(cfg.data, 'train_data_limit', None)
    train_loader, val_loader = get_pretrain_loaders(
        tokenizer_name=cfg.data.tokenizer_name, data_dir=Path(cfg.data.data_dir),
        block_size=cfg.model.max_seq_len, batch_size=cfg.training.batch_size,
        train_data_limit=train_limit, val_data_limit=getattr(cfg.data, 'val_data_limit', None)
    )
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)
    scaler = GradScaler() if cfg.device == 'cuda' and GradScaler else None

    # --- 3. 检查点管理器与加载 ---
    print("\n--- 4. 初始化检查点管理器 ---")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_manager = CheckpointManager(ckpt_dir, model, optimizer, scheduler, scaler)
    start_epoch = 0

    # [核心修改] 统一加载逻辑
    load_ckpt_path = getattr(cfg.training, 'load_from_checkpoint', "none")
    load_only_model = getattr(cfg.training, 'load_only_model', False)

    if load_ckpt_path != "none":
        print(f"检测到加载请求: {load_ckpt_path}")
        # 使用 CheckpointManager 的 load 方法，但传入的是具体路径
        # 注意：resume_from 参数现在是路径或 "latest"/"best"
        start_epoch = ckpt_manager.load(load_ckpt_path, load_only_model=load_only_model)

    # --- 4. 钩子与训练器 ---
    print("--- 1.1. 为模型注册监控钩子 ---")
    hooks = register_hooks(model)
    print(f"✅ 已成功注册 {len(hooks)} 个钩子用于监控内部状态。")
    eff_batch_size = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    print(f"等效批次大小: {eff_batch_size}")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=cfg.device,
        logger=logger, ckpt_manager=ckpt_manager,
        hooks=hooks,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_interval=cfg.logging.log_interval,
        save_interval=cfg.checkpointing.save_interval,
        scaler=scaler,
        clip_grad_norm=cfg.training.clip_grad_norm,
        loss_spike_threshold=cfg.training.loss_spike_threshold,
        max_consecutive_spikes=cfg.training.max_consecutive_spikes,
        grad_norm_history_size=cfg.training.grad_norm_history_size,
        grad_clip_percentile=cfg.training.grad_clip_percentile,
        dynamic_clip_factor=cfg.training.dynamic_clip_factor
    )
    trainer.run(cfg.training.max_epochs, start_epoch)


if __name__ == "__main__":
    main()
# END OF FILE: pretrain/train.py