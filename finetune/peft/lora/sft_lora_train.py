# FILE: finetune/peft/lora/sft_lora_train.py
# -*- coding: utf-8 -*-
"""
[v1.6 - 语义净化] 使用 LoRA 进行SFT的训练主脚本
- 更新脚本以使用新的配置字段名 `base_model_checkpoint`。
"""
import torch
import argparse
from pathlib import Path
import time
import sys
import shutil

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from finetune.sft_data_loader import get_sft_loaders
from pretrain.components.checkpointing import CheckpointManager
from pretrain.components.training_loop import Trainer
from finetune.peft.lora.lora import apply_lora_to_model, freeze_base_model_for_lora

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def main():
    parser = argparse.ArgumentParser(description="[v1.6] [LoRA] 监督微调 (SFT) 脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向SFT LoRA配置YAML文件的路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="启用快速开发运行模式，使用固定名称并清理旧目录")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    base_output_dir = Path(cfg.output_dir)
    if args.fast_dev_run:
        run_name = "fast-dev-run"
        output_dir = base_output_dir / "sft" / "peft" / "lora" / run_name
        if output_dir.exists():
            print(f"🧹 fast_dev_run 模式: 正在清理旧的开发目录 {output_dir}")
            shutil.rmtree(output_dir)
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = cfg.run_name.format(timestamp=timestamp)
        output_dir = base_output_dir / "sft" / "peft" / "lora" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)

    # --- 1. 模型 ---
    cfg.model.use_activation_checkpointing = getattr(cfg.training, 'use_activation_checkpointing', False)
    model = build_model(cfg.model)

    # [核心修改] 读取新的配置字段
    ckpt_path = cfg.sft.base_model_checkpoint
    if args.fast_dev_run:
        pretrain_dev_ckpt_path = base_output_dir / "pretrain" / "fast-dev-run" / "checkpoints" / "ckpt_best.pth"
        print(f"🔩 --fast_dev_run: 自动覆盖检查点加载路径。")
        print(f"   - YAML中路径 (将被忽略): {ckpt_path}")
        print(f"   - 自动解析路径: {pretrain_dev_ckpt_path}")
        ckpt_path = str(pretrain_dev_ckpt_path)

    if ckpt_path and Path(ckpt_path).exists():
        print(f"正在从基础模型检查点加载权重: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ 预训练权重加载成功。")
    else:
        print(f"⚠️ 警告：基础模型检查点 '{ckpt_path}' 未找到。LoRA 将在随机初始化的模型上应用。")

    apply_lora_to_model(
        model,
        rank=cfg.lora.r,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules
    )
    freeze_base_model_for_lora(model)

    model.to(cfg.device)
    print(f"模型已移动到设备: {cfg.device}")

    # --- 2. 数据 ---
    train_loader, val_loader = get_sft_loaders(
        tokenizer_path=Path(cfg.data.tokenizer_name),
        sft_bin_file=Path(cfg.data.sft_data_path),
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size
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
        save_interval=cfg.training.save_interval,
        scaler=scaler,
        clip_grad_norm=cfg.training.clip_grad_norm,
        loss_spike_threshold=cfg.training.loss_spike_threshold,
        max_consecutive_spikes=cfg.training.max_consecutive_spikes,
        grad_norm_history_size=cfg.training.grad_norm_history_size,
        grad_clip_percentile=cfg.training.grad_clip_percentile,
        dynamic_clip_factor=cfg.training.dynamic_clip_factor
    )
    trainer.run(cfg.training.max_epochs, 0)


if __name__ == "__main__":
    main()
# END OF FILE: finetune/peft/lora/sft_lora_train.py