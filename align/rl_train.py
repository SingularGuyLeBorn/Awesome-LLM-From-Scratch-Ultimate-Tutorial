# FILE: align/rl_train.py
"""
[v1.4 - 目录重构版] 通用强化学习 (RL) 训练主脚本。
- 输出目录将根据算法名称自动保存到 runs/rlhf/{algorithm}/ 下。
"""
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import time
import sys
from copy import deepcopy

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path: sys.path.insert(0, project_root)

from utils.config_loader import load_config
from utils.builders import build_model, build_value_model, build_reward_model, build_optimizer, build_scheduler, \
    build_loggers
from models.value_model import ValueModel
from models.reward_model import RewardModel
from align.algorithms.ppo.implementation import compute_advantages, ppo_loss
from align.algorithms.gspo.implementation import gspo_loss
from tqdm import tqdm


def get_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)


class RLExperienceBuffer:
    def __init__(self): self.reset()

    def add(self, rollout_data):
        for key, tensor in rollout_data.items(): self.data[key].append(tensor)

    def get_data(self): return {key: torch.cat(tensors, dim=0) for key, tensors in self.data.items()}

    def reset(self): self.data = {"tokens": [], "log_probs": [], "rewards": [], "end_values": []}


def main():
    parser = argparse.ArgumentParser(description="通用 RL 对齐训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向RL配置YAML文件的路径")
    args = parser.parse_args()

    cfg = load_config(args.config_path, Path(__file__).parent.parent.resolve())

    # [核心修改] 自动创建层级化输出目录
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = cfg.run_name.format(timestamp=timestamp)
    base_output_dir = Path(cfg.output_dir)
    output_dir = base_output_dir / "rlhf" / cfg.rl.algorithm.lower() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, "rl_run")

    policy = build_model(cfg.model).to(cfg.device)
    ref_policy = deepcopy(policy).to(cfg.device)
    value_model = build_value_model(cfg.model).to(cfg.device)
    reward_model = build_reward_model(cfg.model).to(cfg.device)

    for param in ref_policy.parameters(): param.requires_grad = False

    sft_ckpt_path = cfg.rl.load_from_checkpoint
    print(f"\n正在从SFT检查点加载 Policy 和 Value 模型权重: {sft_ckpt_path}")
    sft_checkpoint = torch.load(sft_ckpt_path, map_location=cfg.device)
    policy.load_state_dict(sft_checkpoint['model_state_dict'])
    ref_policy.load_state_dict(sft_checkpoint['model_state_dict'])
    value_model.transformer.load_state_dict(sft_checkpoint['model_state_dict'])
    print("✅ Policy 和 Value 模型权重加载成功。")

    rm_ckpt_path = getattr(cfg.rl, 'reward_model_checkpoint', None)
    if rm_ckpt_path:
        print(f"正在从专用检查点加载 Reward 模型权重: {rm_ckpt_path}")
        rm_checkpoint = torch.load(rm_ckpt_path, map_location=cfg.device)
        reward_model.load_state_dict(rm_checkpoint['model_state_dict'])
        print("✅ Reward 模型权重加载成功。")
    else:
        print("⚠️ 警告: 未指定专用的奖励模型检查点，将使用SFT权重初始化奖励模型。")
        reward_model.transformer.load_state_dict(sft_checkpoint['model_state_dict'])

    policy_optimizer = build_optimizer(policy, cfg.training)
    value_optimizer = build_optimizer(value_model, cfg.training)

    print(f"\n--- 开始 {cfg.rl.algorithm.upper()} 训练 ---")
    experience_buffer = RLExperienceBuffer()

    for epoch in range(cfg.training.max_epochs):
        print(f"\nEpoch {epoch} - Rollout Phase")
        policy.eval();
        value_model.eval();
        reward_model.eval()
        for _ in tqdm(range(cfg.rl.rollout_batches), desc="Rollout"):
            prompts = torch.randint(0, cfg.model.vocab_size, (cfg.training.batch_size, 10), device=cfg.device)
            tokens = prompts
            with torch.no_grad():
                for _ in range(cfg.model.max_seq_len - 10):
                    logits = policy(tokens)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
                    tokens = torch.cat([tokens, next_token], dim=1)

                logits = policy(tokens)
                log_probs = get_log_probs(logits, tokens)
                end_rewards = reward_model(tokens)
                values = value_model(tokens)
                end_values = values[:, -1]

            experience_buffer.add(
                {"tokens": tokens, "log_probs": log_probs, "rewards": end_rewards, "end_values": end_values})

        print(f"Epoch {epoch} - Update Phase")
        policy.train();
        value_model.train()
        all_data = experience_buffer.get_data()

        advantages = compute_advantages(all_data['rewards'], all_data['end_values'])
        returns = advantages + all_data['end_values']

        for _ in tqdm(range(cfg.rl.update_epochs), desc="Update"):
            advantages_broadcasted = advantages.unsqueeze(1).expand(-1, cfg.model.max_seq_len)
            current_logits = policy(all_data['tokens'])
            current_log_probs = get_log_probs(current_logits, all_data['tokens'])

            if cfg.rl.algorithm.lower() in ['ppo', 'grpo']:
                loss = ppo_loss(current_log_probs, all_data['log_probs'], advantages_broadcasted, cfg.rl.clip_epsilon)
            elif cfg.rl.algorithm.lower() == 'gspo':
                loss = gspo_loss(current_log_probs.sum(-1), all_data['log_probs'].sum(-1), advantages,
                                 cfg.rl.clip_epsilon)
            else:
                raise ValueError(f"未知 RL 算法: {cfg.rl.algorithm}")

            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()

            current_values = value_model(all_data['tokens'])
            value_loss = F.mse_loss(current_values, returns.unsqueeze(1).expand_as(current_values))
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        experience_buffer.reset()
        print(f"Epoch {epoch} update finished. Policy Loss: {loss.item():.4f}, Value Loss: {value_loss.item():.4f}")


if __name__ == "__main__":
    main()
# END OF FILE: align/rl_train.py