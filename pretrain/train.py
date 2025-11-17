# FILE: pretrain/train.py
# -*- coding: utf-8 -*-
"""
【v3.9 - DDP 终极稳定版】预训练主脚本。
- 在脚本末尾 cleanup_ddp() 之前增加 barrier()，确保所有进程同步退出。
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

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ddp_utils import setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process

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
    parser = argparse.ArgumentParser(description="[v3.9] 统一预训练/继续预训练脚本 (DDP enabled)")
    parser.add_argument("--config_path", type=str, required=True, help="指向配置YAML文件的路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="启用快速开发运行模式，使用固定名称并清理旧目录")
    args = parser.parse_args()

    setup_ddp()
    world_size = get_world_size()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    output_dir = None
    run_name = ""
    if is_main_process():
        base_output_dir = Path(cfg.output_dir)
        if args.fast_dev_run:
            run_name = "fast-dev-run"
            output_dir = base_output_dir / "pretrain" / run_name
            if output_dir.exists():
                print(f"🧹 fast_dev_run 模式 (主进程): 正在清理旧的开发目录 {output_dir}")
                shutil.rmtree(output_dir)
        else:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            run_name = cfg.run_name.format(timestamp=timestamp)
            output_dir = base_output_dir / "pretrain" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)
    if is_main_process():
        print(f"配置加载自: {args.config_path}")
        print(f"所有输出将保存到: {output_dir}")

    # --- 1. 模型 ---
    cfg.model.use_activation_checkpointing = getattr(cfg.training, 'use_activation_checkpointing', False)
    model = build_model(cfg.model).to(cfg.device)

    if world_size > 1:
        model = DDP(model, device_ids=None if cfg.device == 'cpu' else [int(os.environ["LOCAL_RANK"])])
        print(f"Rank {get_rank()}: 模型已用 DDP 包装。")

    # --- 2. 数据、优化器、调度器、混合精度 ---
    train_limit = getattr(cfg.data, 'train_data_limit', None)
    train_loader, val_loader = get_pretrain_loaders(
        tokenizer_name=cfg.data.tokenizer_name, data_dir=Path(cfg.data.data_dir),
        block_size=cfg.model.max_seq_len, batch_size=cfg.training.batch_size,
        train_data_limit=train_limit, val_data_limit=getattr(cfg.data, 'val_data_limit', None),
        ddp_rank=get_rank(), ddp_world_size=world_size
    )

    model_for_optimizer = model.module if world_size > 1 else model
    optimizer = build_optimizer(model_for_optimizer, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)
    scaler = GradScaler() if cfg.device == 'cuda' and GradScaler else None

    # --- 3. 检查点管理器与加载 ---
    if is_main_process():
        print("\n--- 4. 初始化检查点管理器 (仅主进程) ---")

    ckpt_dir = output_dir / "checkpoints" if is_main_process() else None
    ckpt_manager = CheckpointManager(ckpt_dir, model, optimizer, scheduler, scaler)
    start_epoch = 0

    load_ckpt_path = getattr(cfg.training, 'load_from_checkpoint', "none")
    load_only_model = getattr(cfg.training, 'load_only_model', False)

    if load_ckpt_path != "none":
        if is_main_process():
            print(f"检测到加载请求: {load_ckpt_path}")
        start_epoch = ckpt_manager.load(load_ckpt_path, load_only_model=load_only_model)

    if world_size > 1:
        torch.distributed.barrier()

    # --- 4. 钩子与训练器 ---
    if is_main_process():
        print("--- 1.1. 为模型注册监控钩子 ---")
        hooks = register_hooks(model.module if world_size > 1 else model)
        print(f"✅ 已成功注册 {len(hooks)} 个钩子用于监控内部状态。")
        eff_batch_size = cfg.training.batch_size * cfg.training.gradient_accumulation_steps * world_size
        print(f"全局等效批次大小: {eff_batch_size}")
    else:
        hooks = None

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

    # [CRITICAL DDP FIX] 最终同步点
    # 在主进程可能进行最后的日志/文件操作之前，确保所有进程都已完成其工作。
    # 这可以防止非主进程过早退出，导致主进程在调用 `destroy_process_group` 时挂起。
    if world_size > 1:
        torch.distributed.barrier()

    # --- DDP 清理 ---
    cleanup_ddp()


if __name__ == "__main__":
    main()