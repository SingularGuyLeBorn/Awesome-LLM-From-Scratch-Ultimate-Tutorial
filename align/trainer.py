# FILE: align/trainer.py
# -*- coding: utf-8 -*-
"""
[v2.9 - PPO 算法正确性终极修复版] 统一对齐训练器
- 核心修复: 在 `_update_phase` 中，对 policy_loss, value_loss, kl, entropy 的计算
  全部应用了掩码 (mask)，确保只在有效 token 上进行计算，修复了 "silent bug"。
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

from .algorithms.offline.dpo.implementation import dpo_loss
from .algorithms.offline.orpo.implementation import orpo_loss
from .algorithms.online.ppo.implementation import compute_advantages, ppo_loss
from .algorithms.online.gspo.implementation import gspo_loss
from .algorithms.online.grpo.implementation import grpo_loss
from inference.generate import generate


class RLExperienceBuffer:
    def __init__(self):
        self.reset()

    def add(self, rollout_data: dict):
        for key, tensor in rollout_data.items():
            self.data[key].append(tensor)

    def get_data(self):
        return {key: torch.cat(tensors, dim=0) for key, tensors in self.data.items() if tensors}

    def reset(self):
        self.data = defaultdict(list)


def get_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)


class AlignmentTrainer:
    def __init__(self, cfg, policy_model, reference_model, tokenizer, train_loader, logger, ckpt_manager,
                 value_model=None, reward_model=None, policy_optimizer=None, value_optimizer=None,
                 offline_optimizer=None, offline_scheduler=None):
        self.cfg = cfg
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        self.is_offline = hasattr(cfg, 'offline')
        self.algorithm = cfg.offline.algorithm.lower() if self.is_offline else cfg.rl.algorithm.lower()
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or self.tokenizer.token_to_id("<|endoftext|>")
        self.im_end_id = self.tokenizer.token_to_id("<|im_end|>")

        if self.is_offline:
            self.optimizer = offline_optimizer;
            self.scheduler = offline_scheduler
        else:
            self.value_model = value_model;
            self.reward_model = reward_model
            self.policy_optimizer = policy_optimizer;
            self.value_optimizer = value_optimizer
            self.experience_buffer = RLExperienceBuffer()
            self.group_size = getattr(cfg.rl, 'group_size', 1)

    def train(self):
        print(f"\n--- 开始统一训练器: {self.algorithm.upper()} 模式 ---")
        for epoch in range(self.cfg.training.max_epochs):
            if self.is_offline:
                self._train_offline_epoch(epoch)
            else:
                self._train_online_epoch(epoch)
        self.logger.finish()
        print(f"\n--- {self.algorithm.upper()} 训练完成 ---")

    def _train_offline_epoch(self, epoch):
        global_step = epoch * len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{self.algorithm.upper()} Training]")
        for chosen_tokens, rejected_tokens, _, _ in pbar:
            chosen_tokens, rejected_tokens = chosen_tokens.to(self.cfg.device), rejected_tokens.to(self.cfg.device)
            policy_chosen_logits = self.policy_model(chosen_tokens);
            policy_rejected_logits = self.policy_model(rejected_tokens)
            with torch.no_grad():
                reference_chosen_logits = self.reference_model(chosen_tokens);
                reference_rejected_logits = self.reference_model(rejected_tokens)
            policy_chosen_logps = get_log_probs(policy_chosen_logits, chosen_tokens);
            policy_rejected_logps = get_log_probs(policy_rejected_logits, rejected_tokens)
            reference_chosen_logps = get_log_probs(reference_chosen_logits, chosen_tokens);
            reference_rejected_logps = get_log_probs(reference_rejected_logits, rejected_tokens)
            chosen_im_end_indices = (chosen_tokens == self.im_end_id).nonzero(as_tuple=True);
            rejected_im_end_indices = (rejected_tokens == self.im_end_id).nonzero(as_tuple=True)
            chosen_response_mask = torch.zeros_like(chosen_tokens, dtype=torch.bool);
            rejected_response_mask = torch.zeros_like(rejected_tokens, dtype=torch.bool)
            if len(chosen_im_end_indices[0]) > 0:
                for i in range(chosen_tokens.size(0)): chosen_response_mask[i, chosen_im_end_indices[1][i] + 1:] = True
            if len(rejected_im_end_indices[0]) > 0:
                for i in range(rejected_tokens.size(0)): rejected_response_mask[
                    i, rejected_im_end_indices[1][i] + 1:] = True
            chosen_response_mask &= (chosen_tokens != self.pad_token_id);
            rejected_response_mask &= (rejected_tokens != self.pad_token_id)

            if self.algorithm == 'dpo':
                loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
                                reference_rejected_logps,
                                beta=self.cfg.offline.beta, chosen_mask=chosen_response_mask,
                                rejected_mask=rejected_response_mask,
                                label_smoothing=self.cfg.offline.label_smoothing)
            elif self.algorithm == 'orpo':
                policy_chosen_logps_sum = (policy_chosen_logps * chosen_response_mask).sum(-1);
                policy_rejected_logps_sum = (policy_rejected_logps * rejected_response_mask).sum(-1)
                reference_chosen_logps_sum = (reference_chosen_logps * chosen_response_mask).sum(-1);
                reference_rejected_logps_sum = (reference_rejected_logps * rejected_response_mask).sum(-1)
                loss = orpo_loss(policy_chosen_logits, policy_rejected_logits, policy_chosen_logps_sum,
                                 policy_rejected_logps_sum,
                                 reference_chosen_logps_sum, reference_rejected_logps_sum, chosen_tokens,
                                 self.cfg.offline.alpha)
            else:
                raise ValueError(f"未知的离线算法: {self.algorithm}")
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
            self.scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
            if global_step % self.cfg.logging.log_interval == 0: self.logger.log(
                {f'{self.algorithm}/loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]}, step=global_step)
            global_step += 1

    def _train_online_epoch(self, epoch):
        self._rollout_phase(epoch);
        self._update_phase(epoch)

    def _rollout_phase(self, epoch):
        if self.algorithm == 'ppo':
            self.policy_model.eval();
            self.value_model.eval()
        else:
            self.policy_model.eval()
        self.reward_model.eval()
        prompt_iterator = iter(self.train_loader)

        for _ in tqdm(range(self.cfg.rl.rollout_batches), desc=f"Epoch {epoch} [Rollout]"):
            try:
                prompts = next(prompt_iterator).to(self.cfg.device)
            except StopIteration:
                prompt_iterator = iter(self.train_loader);
                prompts = next(prompt_iterator).to(self.cfg.device)

            group_prompts = prompts.repeat_interleave(self.group_size, dim=0) if self.group_size > 1 else prompts
            max_new_tokens = self.cfg.rl.max_gen_len

            with torch.no_grad():
                tokens = generate(
                    self.policy_model,
                    group_prompts,
                    max_new_tokens=max_new_tokens,
                    eos_id=self.tokenizer.token_to_id("<|endoftext|>")
                )

                target_len = self.cfg.model.max_seq_len
                current_len = tokens.shape[1]
                pad_len = target_len - current_len

                if pad_len < 0:
                    tokens = tokens[:, :target_len]
                    pad_len = 0

                padded_tokens = F.pad(tokens, (0, pad_len), "constant", self.pad_token_id)
                attention_mask = (padded_tokens != self.pad_token_id).long()
                mask = (padded_tokens != self.pad_token_id).float()

                logits = self.policy_model(padded_tokens);
                ref_logits = self.reference_model(padded_tokens)
                log_probs = get_log_probs(logits, padded_tokens);
                ref_log_probs = get_log_probs(ref_logits, padded_tokens)
                end_rewards = self.reward_model(padded_tokens, attention_mask)

                experience_dict = {
                    "tokens": padded_tokens, "log_probs": log_probs,
                    "ref_log_probs": ref_log_probs, "mask": mask
                }

                if self.algorithm == 'ppo':
                    values = self.value_model(padded_tokens)
                    kl = log_probs - ref_log_probs
                    rewards = -self.cfg.rl.kl_coeff * kl

                    true_seq_lengths = (tokens != self.pad_token_id).sum(dim=1)
                    last_token_indices = true_seq_lengths - 1
                    batch_indices = torch.arange(tokens.size(0), device=tokens.device)
                    rewards[batch_indices, last_token_indices] += end_rewards

                    experience_dict["rewards"] = rewards
                    experience_dict["values"] = values
                else:
                    end_rewards_grouped = end_rewards.view(-1, self.group_size)
                    mean_rewards = end_rewards_grouped.mean(dim=1, keepdim=True)
                    std_rewards = end_rewards_grouped.std(dim=1, keepdim=True)
                    advantages_grouped = (end_rewards_grouped - mean_rewards) / (std_rewards + 1e-8)
                    advantages_flat = advantages_grouped.flatten()
                    advantages_unsqueezed = advantages_flat.unsqueeze(1)
                    advantages = advantages_unsqueezed.expand_as(padded_tokens)
                    experience_dict["advantages"] = advantages

                self.experience_buffer.add(experience_dict)

    def _update_phase(self, epoch):
        if self.algorithm == 'ppo':
            self.policy_model.train();
            self.value_model.train()
        else:
            self.policy_model.train()

        all_data = self.experience_buffer.get_data()
        if not all_data:
            print(f"Epoch {epoch} [Update]: 经验缓冲区为空，跳过更新。")
            return

        if self.algorithm == 'ppo':
            advantages, returns = compute_advantages(all_data['rewards'], all_data['values'], all_data['mask'])
            dataset = TensorDataset(all_data['tokens'], all_data['log_probs'], all_data['ref_log_probs'],
                                    all_data['mask'], advantages, returns)
        else:
            dataset = TensorDataset(all_data['tokens'], all_data['log_probs'], all_data['ref_log_probs'],
                                    all_data['mask'], all_data['advantages'])

        dataloader = DataLoader(dataset, batch_size=self.cfg.rl.update_batch_size, shuffle=True)

        for _ in tqdm(range(self.cfg.rl.update_epochs), desc=f"Epoch {epoch} [Update]"):
            for batch in dataloader:
                if self.algorithm == 'ppo':
                    tokens, old_log_probs, ref_log_probs, mask, advantages, returns = [t.to(self.cfg.device) for t in
                                                                                       batch]
                else:
                    tokens, old_log_probs, ref_log_probs, mask, advantages = [t.to(self.cfg.device) for t in batch]

                current_logits = self.policy_model(tokens)
                current_log_probs = get_log_probs(current_logits, tokens)

                if self.algorithm == 'ppo':
                    current_values = self.value_model(tokens)
                    probs = F.softmax(current_logits, dim=-1)
                    log_probs_dist = F.log_softmax(current_logits, dim=-1)

                    # [核心修复] 对 kl 和 entropy 进行掩码平均
                    kl_per_token = current_log_probs - ref_log_probs
                    kl = (kl_per_token * mask).sum() / mask.sum()

                    entropy_per_token = -(probs * log_probs_dist).sum(dim=-1)
                    entropy = (entropy_per_token * mask).sum() / mask.sum()

                    # [核心修复] 将 mask 传递给 ppo_loss
                    policy_loss = ppo_loss(
                        current_log_probs,
                        old_log_probs,
                        advantages,
                        mask,  # <--- 传递掩码
                        self.cfg.rl.clip_epsilon
                    )

                    # [核心修复] 对 value_loss 进行掩码计算
                    value_loss_unmasked = (current_values - returns).pow(2)
                    value_loss_masked = value_loss_unmasked * mask
                    value_loss = value_loss_masked.sum() / mask.sum()

                    total_loss = (
                        policy_loss
                        + self.cfg.rl.kl_coeff * kl
                        - self.cfg.rl.entropy_coef * entropy
                        + self.cfg.rl.value_loss_coef * value_loss
                    )

                    self.policy_optimizer.zero_grad()
                    total_loss.backward()
                    self.policy_optimizer.step()

                elif self.algorithm == 'grpo':
                    policy_loss = grpo_loss(current_log_probs, old_log_probs, advantages, self.cfg.rl.clip_epsilon)
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                elif self.algorithm == 'gspo':
                    policy_loss = gspo_loss(current_log_probs.sum(-1), old_log_probs.sum(-1), advantages.mean(-1),
                                            self.cfg.rl.clip_epsilon)
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

        self.experience_buffer.reset()
        if 'total_loss' in locals():
            print(f"Epoch {epoch} update finished. Last batch stats: Total Loss={total_loss.item():.4f}")
            self.ckpt_manager.save(epoch, total_loss.item())
        elif 'policy_loss' in locals():
            print(f"Epoch {epoch} update finished. Last batch stats: Policy Loss={policy_loss.item():.4f}")
            self.ckpt_manager.save(epoch, policy_loss.item())
# END OF FILE: align/trainer.py