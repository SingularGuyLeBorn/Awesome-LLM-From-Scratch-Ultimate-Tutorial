# FILE: finetune/peft/qlora/sft_qlora_train.py
# -*- coding: utf-8 -*-
"""
[QLoRA Training] 专用的 QLoRA 训练脚本。
更新: 集成自动层名探测，完美支持 MoE 和 MLA。
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
from finetune.peft.qlora.qlora import replace_linear_with_qlora, prepare_model_for_qlora_training
# [新增导入]
from utils.model_utils import find_all_linear_names

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def main():
    parser = argparse.ArgumentParser(description="[QLoRA] 4-bit Quantized SFT Training")
    parser.add_argument("--config_path", type=str, required=True, help="Path to QLoRA config YAML")
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast dev run mode")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    base_output_dir = Path(cfg.output_dir)
    if args.fast_dev_run:
        run_name = "fast-dev-run"
        output_dir = base_output_dir / "sft" / "peft" / "qlora" / run_name
        if output_dir.exists():
            print(f"🧹 fast_dev_run: Cleaning up {output_dir}")
            shutil.rmtree(output_dir)
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = cfg.run_name.format(timestamp=timestamp)
        output_dir = base_output_dir / "sft" / "peft" / "qlora" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_loggers(cfg, output_dir, run_name)

    # --- 1. 模型加载 (FP32/BF16) ---
    print("\n--- 1. Loading Base Model (High Precision) ---")
    cfg.model.use_activation_checkpointing = False
    model = build_model(cfg.model)

    ckpt_path = cfg.sft.base_model_checkpoint
    if args.fast_dev_run:
        pretrain_dev_ckpt_path = base_output_dir / "pretrain" / "fast-dev-run" / "checkpoints" / "ckpt_best.pth"
        print(f"🔩 --fast_dev_run: Overriding checkpoint path -> {pretrain_dev_ckpt_path}")
        ckpt_path = str(pretrain_dev_ckpt_path)

    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Weights loaded to CPU.")
    else:
        print("⚠️ Warning: No checkpoint found. Using random weights (Testing only).")

    # --- 2. QLoRA 转换 (Quantize + Adapt) ---
    print("\n--- 2. Applying QLoRA ---")

    # [核心逻辑] 自动检测 Target Modules
    # 如果 YAML 里没写，或者写了 "auto"，或者我们强制想覆盖
    # 这里我们采取策略：如果 YAML 里有，用 YAML 的；否则自动检测
    # 但为了方便 MoE/MLA 用户，我们建议打印出检测到的 targets

    auto_targets = find_all_linear_names(model)
    print(f"🔍 Auto-detected linear modules: {auto_targets}")

    # 优先使用配置文件中的，如果配置文件未指定，则使用自动检测的
    target_modules = getattr(cfg.qlora, 'target_modules', None)
    if target_modules is None or target_modules == "auto":
        print("   -> Using auto-detected modules for QLoRA.")
        target_modules = auto_targets
    else:
        print(f"   -> Using config specified modules: {target_modules}")

    compute_dtype = torch.bfloat16 if getattr(cfg.qlora, 'compute_dtype', 'float32') == "bfloat16" else torch.float32
    if cfg.device == 'cpu':
        compute_dtype = torch.float32

    replace_linear_with_qlora(
        model,
        rank=cfg.qlora.r,
        alpha=cfg.qlora.alpha,
        dropout=cfg.qlora.dropout,
        target_modules=target_modules,
        compute_dtype=compute_dtype
    )

    prepare_model_for_qlora_training(model)

    print(f"Moving model to {cfg.device}...")
    model.to(cfg.device)

    # --- 3. 数据 ---
    train_loader, val_loader = get_sft_loaders(
        tokenizer_path=Path(cfg.data.tokenizer_name),
        sft_bin_file=Path(cfg.data.sft_data_path),
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size
    )

    # --- 4. 优化器 ---
    optimizer = build_optimizer(model, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)
    scaler = GradScaler() if cfg.device == 'cuda' and GradScaler else None

    # --- 5. 检查点 ---
    ckpt_manager = CheckpointManager(output_dir / "checkpoints", model, optimizer, scheduler, scaler)

    # --- 6. 训练 ---
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
        clip_grad_norm=cfg.training.clip_grad_norm
    )

    print("\n🚀 Starting QLoRA Training...")
    trainer.run(cfg.training.max_epochs, 0)


if __name__ == "__main__":
    main()
# END OF FILE: finetune/peft/qlora/sft_qlora_train.py