# FILE: finetune/sft_lora_train.py
"""
[v1.1 - 最终修复版] 使用 LoRA 进行监督微调 (SFT) 的训练主脚本
- 修复了 Trainer 初始化时参数传递的严重错误。
"""
import torch
import argparse
from pathlib import Path
import time
import sys

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from finetune.sft_data_loader import get_sft_loaders
from pretrain.components.checkpointing import CheckpointManager
from pretrain.components.training_loop import Trainer
from finetune.peft.lora import apply_lora_to_model, freeze_base_model_for_lora

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def main():
    parser = argparse.ArgumentParser(description="[v1.1 修复版] [LoRA] 监督微调 (SFT) 脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向SFT LoRA配置YAML文件的路径")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = cfg.run_name.format(timestamp=timestamp)
    output_dir = Path(cfg.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_loggers(cfg, output_dir, run_name)

    # --- 1. 模型 ---
    model = build_model(cfg.model)

    if cfg.sft.load_from_checkpoint:
        print(f"正在从检查点加载预训练权重: {cfg.sft.load_from_checkpoint}")
        checkpoint = torch.load(cfg.sft.load_from_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ 预训练权重加载成功。")

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
    # [核心修复] 改为清晰、明确的参数传递
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
# END OF FILE: finetune/sft_lora_train.py