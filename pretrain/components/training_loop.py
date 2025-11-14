# FILE: pretrain/components/training_loop.py
"""
【遥测版】实现核心的训练和评估循环。
集成了钩子，用于监控内部激活值和梯度。
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Optional, List

from .logging import Logger
from .checkpointing import CheckpointManager
from evaluation.metrics.perplexity import calculate_perplexity
from .hooks import ForwardHook, BackwardHook  # 导入钩子

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device,
                 gradient_accumulation_steps: int, clip_grad_norm: float,
                 logger: Logger, ckpt_manager: CheckpointManager,
                 log_interval: int, save_interval: int,
                 scaler: Optional[GradScaler] = None,
                 hooks: Optional[List] = None):  # <-- 接收钩子列表
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.scaler = scaler
        self.hooks = hooks  # <-- 存储钩子

        self.use_cpu_amp = (self.device == 'cpu')
        self.use_gpu_amp = (self.device == 'cuda' and self.scaler is not None)

        if self.use_cpu_amp:
            logging.info("Device is CPU. Automatic mixed precision with bfloat16 will be attempted.")
        if self.use_gpu_amp:
            logging.info("Device is CUDA. Using automatic mixed precision (fp16) with GradScaler.")

    def _compute_loss(self, logits, y, loss_mask):
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss_mask = loss_mask.view(-1)
        loss = F.cross_entropy(logits, y, reduction='none')
        masked_loss = loss * loss_mask
        avg_loss = masked_loss.sum() / (loss_mask.sum() + 1e-9)
        return avg_loss

    def _collect_hook_metrics(self) -> dict:
        """从所有钩子中收集指标。"""
        metrics = {}
        if self.hooks:
            for hook in self.hooks:
                metrics.update(hook.get_metric())
        return metrics

    def train_epoch(self, epoch, global_step):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Training]")
        for i, (x, y, loss_mask) in enumerate(pbar):
            x, y, loss_mask = x.to(self.device), y.to(self.device), loss_mask.to(self.device)

            amp_context = torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.use_cpu_amp else torch.float16,
                enabled=(self.use_cpu_amp or self.use_gpu_amp)
            )

            with amp_context:
                logits = self.model(x)
                loss = self._compute_loss(logits, y, loss_mask)
                loss = loss / self.gradient_accumulation_steps

            if self.use_gpu_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.use_gpu_amp: self.scaler.unscale_(self.optimizer)
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                if self.use_gpu_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % self.log_interval == 0:
                    metrics_to_log = {
                        'train/loss_step': loss.item() * self.gradient_accumulation_steps,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    # --- 核心修改：收集并添加钩子指标 ---
                    hook_metrics = self._collect_hook_metrics()
                    metrics_to_log.update(hook_metrics)
                    self.logger.log(metrics_to_log, step=global_step)

                if global_step % self.save_interval == 0:
                    self.ckpt_manager.save(epoch, -1, is_best=False)

            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix(loss=loss.item() * self.gradient_accumulation_steps, step=global_step)

        return total_loss / len(self.train_loader), global_step

    @torch.no_grad()
    def evaluate(self, epoch, global_step):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Validation]")
        for x, y, loss_mask in pbar:
            x, y, loss_mask = x.to(self.device), y.to(self.device), loss_mask.to(self.device)
            amp_context = torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.use_cpu_amp else torch.float16,
                enabled=(self.use_cpu_amp or self.use_gpu_amp)
            )
            with amp_context:
                logits = self.model(x)
                loss = self._compute_loss(logits, y, loss_mask)
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        perplexity = calculate_perplexity(avg_loss)
        return {'loss': avg_loss, 'perplexity': perplexity}

    def run(self, max_epochs: int, start_epoch: int):
        logging.info("--- 开始训练 ---")
        global_step = start_epoch * len(self.train_loader)

        for epoch in range(start_epoch, max_epochs):
            avg_train_loss, global_step = self.train_epoch(epoch, global_step)
            eval_metrics = self.evaluate(epoch, global_step)

            summary_metrics = {
                'train/loss_epoch': avg_train_loss,
                'val/loss': eval_metrics['loss'],
                'val/perplexity': eval_metrics['perplexity'],
            }
            self.logger.log(summary_metrics, step=epoch)

            is_best = eval_metrics['loss'] < self.ckpt_manager.best_val_loss
            if is_best: self.ckpt_manager.best_val_loss = eval_metrics['loss']
            self.ckpt_manager.save(epoch, eval_metrics['loss'], is_best)

        self.logger.finish()
        logging.info("--- 训练完成 ---")

# END OF FILE: pretrain/components/training_loop.py