# FILE: align/train_offline.py
# -*- coding: utf-8 -*-
"""
[v3.1 - 修复版] 通用离线对齐训练脚本 (DPO, ORPO, etc.)
- 修复了调用 get_preference_loaders 时缺少 tokenizer_name 参数的错误。
"""
import argparse
from pathlib import Path
import time
import sys
from copy import deepcopy
from tokenizers import Tokenizer

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path: sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from align.preference_data_loader import get_preference_loaders
from pretrain.components.checkpointing import CheckpointManager
from align.trainer import AlignmentTrainer
import torch


def main():
    parser = argparse.ArgumentParser(description="[v3.1] 通用离线对齐训练启动器")
    parser.add_argument("--config_path", type=str, required=True, help="指向离线对齐配置YAML文件的路径")
    args = parser.parse_args()

    # --- 1. 加载配置和初始化 ---
    cfg = load_config(args.config_path, Path(__file__).parent.parent.resolve())
    algorithm = cfg.offline.algorithm.lower()
    output_dir = Path(
        cfg.output_dir) / "rlhf" / "offline" / f"{cfg.run_name.format(timestamp=time.strftime('%Y%m%d-%H%M%S'))}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, f"{algorithm}_run")
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_name)

    # --- 2. 构建模型 ---
    print("\n--- 构建 Policy 和 Reference 模型 ---")
    policy_model = build_model(cfg.model).to(cfg.device)
    reference_model = deepcopy(policy_model).to(cfg.device)
    for param in reference_model.parameters():
        param.requires_grad = False

    print(f"正在从SFT检查点加载权重: {cfg.offline.load_from_checkpoint}")
    checkpoint = torch.load(cfg.offline.load_from_checkpoint, map_location=cfg.device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    reference_model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Policy 和 Reference 模型权重加载成功。")

    # --- 3. 构建数据加载器 ---
    print("\n--- 构建偏好数据加载器 ---")
    # [核心修复] 将 tokenizer_name 传递给 get_preference_loaders
    train_loader = get_preference_loaders(
        data_dir=Path(cfg.data.data_dir),
        tokenizer_name=cfg.data.tokenizer_name,
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size
    )

    # --- 4. 构建优化器和调度器 ---
    print("\n--- 构建优化器和调度器 ---")
    optimizer = build_optimizer(policy_model, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)

    # --- 5. 构建检查点管理器 ---
    ckpt_manager = CheckpointManager(output_dir / "checkpoints", policy_model, optimizer, scheduler)

    # --- 6. 实例化并运行 AlignmentTrainer ---
    trainer = AlignmentTrainer(
        cfg=cfg,
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        logger=logger,
        ckpt_manager=ckpt_manager,
        offline_optimizer=optimizer,
        offline_scheduler=scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
# END OF FILE: align/train_offline.py