# FILE: align/rm_train.py
"""
[v1.1 - 目录重构版] 奖励模型 (Reward Model, RM) 训练主脚本。
- 输出目录将自动保存到 runs/rlhf/rm/ 下。
"""
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import time
import sys

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_reward_model, build_optimizer, build_scheduler, build_loggers
from align.preference_data_loader import get_preference_loaders
from tqdm import tqdm


def rm_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def main():
    parser = argparse.ArgumentParser(description="奖励模型 (RM) 训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向RM配置YAML文件的路径")
    args = parser.parse_args()

    # --- 0. 配置与日志 ---
    cfg = load_config(args.config_path, Path(__file__).parent.parent.resolve())

    # [核心修改] 自动创建层级化输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = cfg.run_name.format(timestamp=timestamp)
    base_output_dir = Path(cfg.output_dir)
    output_dir = base_output_dir / "rlhf" / "rm" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, "rm_run")

    # --- 1. 初始化模型 ---
    reward_model = build_reward_model(cfg.model).to(cfg.device)

    if cfg.rm.load_from_checkpoint:
        print(f"正在从SFT检查点加载权重: {cfg.rm.load_from_checkpoint}")
        checkpoint = torch.load(cfg.rm.load_from_checkpoint, map_location=cfg.device)
        reward_model.transformer.load_state_dict(checkpoint['model_state_dict'])
        print("✅ RM Transformer 权重加载成功。")

    # --- 2. 数据 ---
    train_loader = get_preference_loaders(
        data_dir=Path(cfg.data.data_dir),
        block_size=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size
    )

    # --- 3. 优化器与调度器 ---
    optimizer = build_optimizer(reward_model, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)

    # --- 4. 训练循环 ---
    print("\n--- 开始奖励模型 (RM) 训练 ---")
    global_step = 0
    for epoch in range(cfg.training.max_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [RM Training]")
        for chosen_tokens, rejected_tokens in pbar:
            chosen_tokens, rejected_tokens = chosen_tokens.to(cfg.device), rejected_tokens.to(cfg.device)

            chosen_rewards = reward_model(chosen_tokens)
            rejected_rewards = reward_model(rejected_tokens)

            loss = rm_loss(chosen_rewards, rejected_rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.2f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % cfg.logging.log_interval == 0:
                logger.log({'rm/loss': loss.item(), 'rm/accuracy': accuracy, 'lr': scheduler.get_last_lr()[0]},
                           step=global_step)

            global_step += 1

    logger.finish()
    print("\n--- 奖励模型训练完成 ---")

    final_ckpt_path = output_dir / "rm_final.pth"
    torch.save({'model_state_dict': reward_model.state_dict()}, final_ckpt_path)
    print(f"✅ 最终奖励模型已保存到: {final_ckpt_path}")


if __name__ == "__main__":
    main()
# END OF FILE: align/rm_train.py