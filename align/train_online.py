# FILE: align/train_online.py
# -*- coding: utf-8 -*-
"""
[v3.1 - 理论完备版启动器] 通用在线 RL 对齐训练脚本 (PPO, GSPO, etc.)
- 为PPO创建单一优化器，以支持统一损失函数。
"""
import argparse
from pathlib import Path
import time
import sys
from tokenizers import Tokenizer
import torch

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import (
    build_model, build_value_model, build_reward_model,
    build_optimizer, build_loggers
)
from align.prompt_loader import get_prompt_loader
from pretrain.components.checkpointing import CheckpointManager
from align.trainer import AlignmentTrainer


def log_run_details(cfg, output_dir):
    group_size = getattr(cfg.rl, 'group_size', 1)
    prompt_file_path = Path(cfg.data.prompt_data_path)
    if prompt_file_path.exists():
        num_prompts = len(prompt_file_path.read_text(encoding='utf-8').strip().split('\n'))
        prompt_loader_len = (num_prompts + cfg.training.batch_size - 1) // cfg.training.batch_size
    else:
        prompt_loader_len = 0
    rollout_batches = min(cfg.rl.rollout_batches,
                          prompt_loader_len) if prompt_loader_len > 0 else cfg.rl.rollout_batches
    total_samples_per_epoch = rollout_batches * cfg.training.batch_size * group_size

    print("\n" + "=" * 50 + f"\n{'RLHF 训练参数详情':^50}\n" + "=" * 50)
    print(f" ▶ 运行名称: {Path(output_dir).name}\n ▶ 输出目录: {output_dir}\n ▶ 使用算法: {cfg.rl.algorithm.upper()}")
    print("-" * 50 + f"\n{'核心训练参数':^50}\n" + "-" * 50)
    print(
        f"   - 学习率: {cfg.training.learning_rate:.2e}\n   - 批次大小 (Prompt): {cfg.training.batch_size}\n   - 总 Epochs: {cfg.training.max_epochs}")
    print("-" * 50 + f"\n{'RL 核心参数':^50}\n" + "-" * 50)
    print(
        f"   - Rollout 批次数/Epoch: {rollout_batches}\n   - Update 次数/Epoch: {cfg.rl.update_epochs}\n   - PPO 裁剪 Epsilon: {cfg.rl.clip_epsilon}")
    print(f"   - KL 惩罚系数: {cfg.rl.kl_coeff}")
    if group_size > 1: print(f"   - 组大小 (Group Size): {group_size}")
    print(f" ℹ 总计每个 Epoch 将生成 {total_samples_per_epoch} 个样本序列。")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="[v3.1] 通用在线 RL 对齐训练启动器")
    parser.add_argument("--config_path", type=str, required=True, help="指向RL配置YAML文件的路径")
    args = parser.parse_args()

    cfg = load_config(args.config_path, Path(__file__).parent.parent.resolve())
    algorithm = cfg.rl.algorithm.lower()
    output_dir = Path(
        cfg.output_dir) / "rlhf" / "online" / f"{cfg.run_name.format(timestamp=time.strftime('%Y%m%d-%H%M%S'))}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, "rl_run")
    log_run_details(cfg, output_dir)
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_name)

    print("\n--- 1. 初始化所有模型 ---")
    policy_model = build_model(cfg.model).to(cfg.device)
    reference_model = build_model(cfg.model).to(cfg.device)
    value_model = build_value_model(cfg.model).to(cfg.device) if algorithm == 'ppo' else None
    reward_model = build_reward_model(cfg.model).to(cfg.device)

    print("\n--- 2. 加载检查点权重 ---")
    sft_ckpt = torch.load(cfg.rl.load_from_checkpoint, map_location=cfg.device)
    policy_model.load_state_dict(sft_ckpt['model_state_dict'])
    reference_model.load_state_dict(sft_ckpt['model_state_dict'])
    if value_model:
        value_model.transformer.load_state_dict(sft_ckpt['model_state_dict'])
    print(f"✅ Policy, Reference, Value (if any) 模型已从 SFT 检查点 '{cfg.rl.load_from_checkpoint}' 加载。")

    rm_ckpt = torch.load(cfg.rl.reward_model_checkpoint, map_location=cfg.device)
    reward_model.load_state_dict(rm_ckpt['model_state_dict'])
    print(f"✅ Reward 模型已从专用检查点 '{cfg.rl.reward_model_checkpoint}' 加载。")

    for param in reference_model.parameters():
        param.requires_grad = False

    print("\n--- 3. 初始化 Prompt 数据加载器 ---")
    prompt_loader = get_prompt_loader(
        prompt_file_path=Path(cfg.data.prompt_data_path),
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        max_prompt_len=cfg.rl.max_prompt_len
    )

    print("\n--- 4. 初始化优化器 ---")
    policy_optimizer, value_optimizer = None, None
    if algorithm == 'ppo':
        # [核心修改] PPO 使用单一优化器管理 policy 和 value model
        params_to_optimize = list(policy_model.parameters()) + list(value_model.parameters())
        # 创建一个临时的SimpleNamespace来传递给build_optimizer
        ppo_optim_config = lambda: None
        setattr(ppo_optim_config, 'learning_rate', cfg.training.learning_rate)
        setattr(ppo_optim_config, 'weight_decay', cfg.training.weight_decay)
        # 合并模型参数并创建单一优化器
        combined_model = torch.nn.ModuleList([policy_model, value_model])
        policy_optimizer = build_optimizer(combined_model, ppo_optim_config)
        value_optimizer = None  # 不再需要独立的 value_optimizer
    else:  # GRPO, GSPO
        policy_optimizer = build_optimizer(policy_model, cfg.training)
        value_optimizer = None

    print("\n--- 5. 构建检查点管理器 ---")
    ckpt_manager = CheckpointManager(output_dir / "checkpoints", policy_model, policy_optimizer, scheduler=None)

    trainer = AlignmentTrainer(
        cfg=cfg,
        policy_model=policy_model,
        reference_model=reference_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        train_loader=prompt_loader,
        logger=logger,
        ckpt_manager=ckpt_manager,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,  # 传递None给GRPO/GSPO
    )

    trainer.train()


if __name__ == "__main__":
    main()

