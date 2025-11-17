# FILE: align/train_online.py
# -*- coding: utf-8 -*-
"""
[v3.4 - 语义净化] 通用在线 RL 对齐训练脚本 (PPO, GSPO, etc.)
- 更新脚本以使用新的配置字段名 `sft_model_checkpoint`。
"""
import argparse
from pathlib import Path
import time
import sys
from tokenizers import Tokenizer
import torch
import shutil

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
    parser = argparse.ArgumentParser(description="[v3.4] 通用在线 RL 对齐训练启动器")
    parser.add_argument("--config_path", type=str, required=True, help="指向RL配置YAML文件的路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="启用快速开发运行模式，使用固定名称并清理旧目录")
    args = parser.parse_args()

    cfg = load_config(args.config_path, Path(__file__).parent.parent.resolve())
    algorithm = cfg.rl.algorithm.lower()

    base_output_dir = Path(cfg.output_dir)
    if args.fast_dev_run:
        run_name = "fast-dev-run"
        output_dir = base_output_dir / "rlhf" / "online" / f"{algorithm}-{run_name}"
        if output_dir.exists():
            print(f"🧹 fast_dev_run 模式: 正在清理旧的开发目录 {output_dir}")
            shutil.rmtree(output_dir)
    else:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = cfg.run_name.format(timestamp=timestamp)
        output_dir = base_output_dir / "rlhf" / "online" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, "rl_run")
    log_run_details(cfg, output_dir)
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_name)

    print("\n--- 1. 初始化所有模型 ---")
    cfg.model.use_activation_checkpointing = getattr(cfg.training, 'use_activation_checkpointing', False)
    policy_model = build_model(cfg.model).to(cfg.device)
    reference_model = build_model(cfg.model).to(cfg.device)
    value_model = build_value_model(cfg.model).to(cfg.device) if algorithm == 'ppo' else None
    reward_model = build_reward_model(cfg.model).to(cfg.device)

    print("\n--- 2. 加载检查点权重 ---")
    # [核心修改] 读取新的配置字段
    sft_ckpt_path = cfg.rl.sft_model_checkpoint
    rm_ckpt_path = cfg.rl.reward_model_checkpoint
    if args.fast_dev_run:
        sft_dev_ckpt_path = base_output_dir / "sft" / "full" / "fast-dev-run" / "checkpoints" / "ckpt_best.pth"
        rm_dev_ckpt_path = base_output_dir / "rlhf" / "rm" / "fast-dev-run" / "checkpoints" / "ckpt_best.pth"
        print(f"🔩 --fast_dev_run: 自动覆盖SFT和RM检查点加载路径。")
        sft_ckpt_path = str(sft_dev_ckpt_path)
        rm_ckpt_path = str(rm_dev_ckpt_path)
        print(f"   - SFT Ckpt -> {sft_ckpt_path}")
        print(f"   - RM Ckpt  -> {rm_ckpt_path}")

    # 加载 SFT 检查点
    if sft_ckpt_path and Path(sft_ckpt_path).exists():
        sft_ckpt = torch.load(sft_ckpt_path, map_location=cfg.device)
        policy_model.load_state_dict(sft_ckpt['model_state_dict'])
        reference_model.load_state_dict(sft_ckpt['model_state_dict'])
        if value_model:
            value_model.transformer.load_state_dict(sft_ckpt['model_state_dict'])
        print(f"✅ Policy, Reference, Value (if any) 模型已从 SFT 检查点 '{sft_ckpt_path}' 加载。")
    else:
        print(f"⚠️ 警告: SFT 检查点 '{sft_ckpt_path}' 未找到，模型将使用随机权重。")

    # 加载 RM 检查点
    if rm_ckpt_path and Path(rm_ckpt_path).exists():
        rm_ckpt = torch.load(rm_ckpt_path, map_location=cfg.device)
        reward_model.load_state_dict(rm_ckpt['model_state_dict'])
        print(f"✅ Reward 模型已从专用检查点 '{rm_ckpt_path}' 加载。")
    else:
        print(f"⚠️ 警告: 奖励模型检查点 '{rm_ckpt_path}' 未找到，奖励模型将使用随机权重。")

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
        params_to_optimize = list(policy_model.parameters()) + list(value_model.parameters())
        ppo_optim_config = lambda: None
        setattr(ppo_optim_config, 'learning_rate', cfg.training.learning_rate)
        setattr(ppo_optim_config, 'weight_decay', cfg.training.weight_decay)
        combined_model = torch.nn.ModuleList([policy_model, value_model])
        policy_optimizer = build_optimizer(combined_model, ppo_optim_config)
        value_optimizer = None
    else:
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
        value_optimizer=value_optimizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
# END OF FILE: align/train_online.py