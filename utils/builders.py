# FILE: utils/builders.py
"""
[v1.5 - 架构统一版] 组件构建器模块。
- 新增 build_value_model 和 build_reward_model，确保所有模型都通过正确的逻辑创建。
"""
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace

from models.transformer import Transformer, ModelArgs
from models.value_model import ValueModel
from models.reward_model import RewardModel
from pretrain.components.optimizer import get_optimizer
from pretrain.components.scheduler import get_lr_scheduler
from pretrain.components.logging import (
    Logger, ConsoleLogger, FileLogger, HumanReadableLogger,
    WandbLogger, SwanlabLogger
)

def build_model(model_config: SimpleNamespace) -> nn.Module:
    """根据配置构建基础 Transformer 模型。"""
    print("\n--- 1. 初始化模型 ---")
    model_args = ModelArgs(**vars(model_config))
    model = Transformer(model_args)
    print(f"基础 Transformer 模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model

def build_value_model(model_config: SimpleNamespace) -> nn.Module:
    """[新增] 根据配置构建 ValueModel。"""
    print("--- 1.1. 初始化价值模型 (Critic) ---")
    model_args = ModelArgs(**vars(model_config))
    model = ValueModel(model_args)
    print(f"价值模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model

def build_reward_model(model_config: SimpleNamespace) -> nn.Module:
    """[新增] 根据配置构建 RewardModel。"""
    print("--- 1.2. 初始化奖励模型 ---")
    model_args = ModelArgs(**vars(model_config))
    model = RewardModel(model_args)
    print(f"奖励模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model

def build_optimizer(model: nn.Module, train_config: SimpleNamespace) -> torch.optim.Optimizer:
    print("\n--- 3. 初始化优化器 ---")
    optimizer = get_optimizer(model, train_config.learning_rate, train_config.weight_decay)
    print(f"✅ 优化器构建完成 (类型: {type(optimizer).__name__})")
    return optimizer

def build_scheduler(optimizer: torch.optim.Optimizer, train_config: SimpleNamespace, max_iters: int) -> torch.optim.lr_scheduler._LRScheduler:
    print("--- 3.1. 初始化调度器 ---")
    warmup_iters = int(max_iters * getattr(train_config, 'warmup_ratio', 0))
    min_lr = train_config.learning_rate * getattr(train_config, 'min_lr_ratio', 0)
    scheduler = get_lr_scheduler(optimizer, warmup_iters, max_iters, min_lr)
    print(f"✅ 学习率调度器构建完成 (类型: CosineAnnealing with Linear Warmup)")
    print(f"   - 总步数: {max_iters}, 预热步数: {warmup_iters}")
    return scheduler

def build_loggers(config: SimpleNamespace, output_dir: Path, run_name: str) -> Logger:
    # (此函数保持不变)
    print("\n--- 0. 初始化日志系统 ---")
    loggers_to_use = []
    console_cfg = getattr(config, 'console', SimpleNamespace(verbose=False))
    loggers_to_use.append(ConsoleLogger(**vars(console_cfg)))
    loggers_to_use.append(FileLogger(log_dir=output_dir))
    loggers_to_use.append(HumanReadableLogger(log_dir=output_dir))
    logging_cfg = getattr(config, 'logging', SimpleNamespace())
    wandb_cfg = getattr(logging_cfg, 'wandb', SimpleNamespace(enable=False))
    if wandb_cfg and wandb_cfg.enable:
        try:
            loggers_to_use.append(WandbLogger(project=wandb_cfg.project, run_name=run_name, log_dir=output_dir, config=vars(config)))
        except ImportError as e:
            print(f"WARNING: 无法添加WandbLogger: {e}.")
    swanlab_cfg = getattr(logging_cfg, 'swanlab', SimpleNamespace(enable=False))
    if swanlab_cfg and swanlab_cfg.enable:
        try:
            exp_name = getattr(swanlab_cfg, 'experiment_name', run_name).format(run_name=run_name)
            loggers_to_use.append(SwanlabLogger(project=swanlab_cfg.project, experiment_name=exp_name, log_dir=output_dir, config=vars(config)))
        except ImportError as e:
            print(f"WARNING: 无法添加SwanlabLogger: {e}.")
    return Logger(loggers_to_use)
# END OF FILE: utils/builders.py