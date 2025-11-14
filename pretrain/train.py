# FILE: pretrain/train.py
"""
【v3 - 健壮版】预训练主脚本。
- 修复了 get_loaders 的调用错误。
- 恢复了所有注释和清晰的代码结构。
"""
import torch
import argparse
import logging
from pathlib import Path
import time
import yaml
from types import SimpleNamespace
import os
import sys

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 路径修复结束 ---

from models.transformer import Transformer
from models.config import ModelArgs
from pretrain.data_loader import get_loaders
from pretrain.components.optimizer import get_optimizer
from pretrain.components.scheduler import get_lr_scheduler
from pretrain.components.logging import Logger, FileLogger, WandbLogger, SwanlabLogger
from pretrain.components.checkpointing import CheckpointManager
from pretrain.components.training_loop import Trainer
from pretrain.components.hooks import register_hooks

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def load_config(config_path: str) -> SimpleNamespace:
    """
    加载YAML配置文件，解析相对路径，并转换为SimpleNamespace对象。
    """
    config_path = Path(config_path)
    # 以配置文件所在的目录为基准，来解析所有相对路径
    base_path = config_path.parent
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 递归地解析字典中所有可能是相对路径的字段
    def resolve_paths_in_dict(d, base):
        for k, v in d.items():
            if isinstance(v, dict):
                resolve_paths_in_dict(v, base)
            # 约定：所有以 _dir, _path, 或名为 tokenizer_name 的字段都可能是路径
            elif isinstance(v, str) and ("_dir" in k or "_path" in k or "tokenizer_name" in k):
                # 如果值不含路径分隔符，且不以.json结尾，则很可能是HF Hub名，不进行解析
                if not os.path.sep in v and not Path(v).suffix == '.json':
                    continue

                resolved = (base / v).resolve()
                # 只有当路径实际存在时，才更新字典中的值
                if resolved.exists():
                    d[k] = str(resolved)

    resolve_paths_in_dict(config_dict, base_path)

    # 递归地将字典转换为SimpleNamespace，方便用点语法访问
    def dict_to_sns(d):
        if isinstance(d, dict): return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
        return d

    return dict_to_sns(config_dict)


def main():
    parser = argparse.ArgumentParser(description="从零手写LLM预训练脚本 (v3 - 健壮版)")
    parser.add_argument("--config_path", type=str, required=True, help="指向预训练配置YAML文件的路径")
    args = parser.parse_args()

    # --- 0. 加载配置并初始化日志 ---
    cfg = load_config(args.config_path)

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = cfg.run_name.format(timestamp=timestamp)
    output_dir = Path(cfg.output_dir) / run_name

    # --- 恢复了清晰的日志初始化结构 ---
    # 首先，总是初始化一个本地文件日志记录器
    loggers_to_use = [FileLogger(log_dir=output_dir)]

    # 根据配置，有条件地添加WandB日志记录器
    if hasattr(cfg.logging, 'wandb') and cfg.logging.wandb.enable:
        try:
            wandb_logger = WandbLogger(
                project=cfg.logging.wandb.project,
                run_name=run_name,
                log_dir=output_dir,
                config=vars(cfg)
            )
            loggers_to_use.append(wandb_logger)
        except ImportError as e:
            logging.warning(f"无法添加WandbLogger: {e}. 请确保'wandb'已安装。")

    # 根据配置，有条件地添加SwanLab日志记录器
    if hasattr(cfg.logging, 'swanlab') and cfg.logging.swanlab.enable:
        try:
            swanlab_exp_name = getattr(cfg.logging.swanlab, 'experiment_name', run_name).format(run_name=run_name)
            swanlab_logger = SwanlabLogger(
                project=cfg.logging.swanlab.project,
                experiment_name=swanlab_exp_name,
                log_dir=output_dir,
                config=vars(cfg)
            )
            loggers_to_use.append(swanlab_logger)
        except ImportError as e:
            logging.warning(f"无法添加SwanlabLogger: {e}. 请确保'swanlab'已安装。")

    # 初始化主日志调度器
    logger = Logger(loggers=loggers_to_use)

    logging.info(f"配置加载自: {args.config_path}")
    logging.info(f"所有输出将保存到: {output_dir}")

    # --- 1. 初始化模型 ---
    logging.info("--- 1. 初始化模型 ---")
    model_args = ModelArgs(**vars(cfg.model))
    model = Transformer(model_args).to(cfg.device)

    logging.info("--- 1.1. 为模型注册监控钩子 ---")
    hooks = register_hooks(model)
    logging.info(f"已成功注册 {len(hooks)} 个钩子用于监控内部状态。")

    effective_batch_size = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    logging.info(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    logging.info(f"等效批次大小: {effective_batch_size}")

    # --- 2. 准备数据加载器 ---
    logging.info("--- 2. 准备数据加载器 (Packed Sequences) ---")
    # 【核心修复】调用get_loaders时，不再传递已废弃的train_filename等参数。
    # DataLoader现在只依赖于data_dir和tokenizer_name，因为它假定.bin文件已由预处理脚本生成。
    train_loader, val_loader = get_loaders(
        tokenizer_name=cfg.data.tokenizer_name,
        data_dir=Path(cfg.data.data_dir),
        block_size=model_args.max_seq_len,
        batch_size=cfg.training.batch_size
    )

    # --- 3. 初始化优化器、调度器和混合精度 ---
    logging.info("--- 3. 初始化优化器、调度器和混合精度 ---")
    optimizer = get_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
    max_iters = len(train_loader) * cfg.training.max_epochs
    warmup_iters = int(max_iters * cfg.training.warmup_ratio)
    min_lr = cfg.training.learning_rate * cfg.training.min_lr_ratio
    scheduler = get_lr_scheduler(optimizer, warmup_iters, max_iters, min_lr)
    scaler = None
    if cfg.device == 'cuda' and GradScaler is not None:
        scaler = GradScaler()
        logging.info("使用CUDA，已初始化GradScaler。")

    # --- 4. 初始化检查点管理器 ---
    logging.info("--- 4. 初始化检查点管理器 ---")
    ckpt_manager = CheckpointManager(output_dir, model, optimizer, scheduler, scaler)
    start_epoch = 0
    if cfg.checkpointing.resume_from != "none":
        start_epoch = ckpt_manager.load(cfg.checkpointing.resume_from)

    # --- 5. 初始化训练器 ---
    logging.info("--- 5. 初始化训练器 ---")
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=cfg.device,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        clip_grad_norm=cfg.training.clip_grad_norm,
        logger=logger, ckpt_manager=ckpt_manager,
        log_interval=cfg.logging.log_interval, save_interval=cfg.checkpointing.save_interval,
        scaler=scaler,
        hooks=hooks
    )

    # --- 6. 启动训练 ---
    trainer.run(cfg.training.max_epochs, start_epoch)


if __name__ == "__main__":
    main()
# END OF FILE: pretrain/train.py