# FILE: pretrain/components/training_loop.py
"""
【v3.7 - DDP 死锁终极修复版】
- 重构 evaluate 方法，让所有 DDP 进程参与数据加载迭代，但只有主进程计算，彻底修复验证死锁。
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Optional, List
from collections import deque
import time
import os

try:
    import psutil
except ImportError:
    psutil = None

from .logging import Logger
from .checkpointing import CheckpointManager
from evaluation.metrics.perplexity import calculate_perplexity
from .hooks import ForwardHook, BackwardHook
from utils.ddp_utils import is_main_process, get_world_size

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device,
                 gradient_accumulation_steps: int,
                 logger: Logger, ckpt_manager: CheckpointManager,
                 log_interval: int, save_interval: int,
                 scaler: Optional[GradScaler] = None,
                 hooks: Optional[List] = None,
                 clip_grad_norm: float = 1.0,
                 loss_spike_threshold: float = 5.0,
                 max_consecutive_spikes: int = 5,
                 grad_norm_history_size: int = 100,
                 grad_clip_percentile: float = 0.9,
                 dynamic_clip_factor: float = 1.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logger = logger
        self.ckpt_manager = ckpt_manager
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.scaler = scaler
        self.hooks = hooks

        self.last_train_loss = None
        self.loss_spike_threshold = loss_spike_threshold
        self.consecutive_spikes = 0
        self.max_consecutive_spikes = max_consecutive_spikes

        self.grad_norm_history = deque(maxlen=grad_norm_history_size)
        self.grad_clip_percentile = grad_clip_percentile
        self.dynamic_clip_factor = dynamic_clip_factor
        self.static_clip_grad_norm = clip_grad_norm

        self.use_cpu_amp = (device == 'cpu')
        self.use_gpu_amp = (device == 'cuda' and scaler is not None)
        self.process = psutil.Process(os.getpid()) if psutil else None

        if is_main_process():
            print("\n--- 5. 初始化训练器 ---")
            print("✅ 训练器构建完成，配置如下:")
            print(f"   - 设备: {self.device}")
            print(f"   - 混合精度 (CPU bfloat16): {'启用' if self.use_cpu_amp else '禁用'}")
            print(f"   - 混合精度 (GPU float16): {'启用' if self.use_gpu_amp else '禁用'}")
            print(f"   - 梯度累积步数: {self.gradient_accumulation_steps}")
            print(f"   - DDP World Size: {get_world_size()}")

    def _compute_loss(self, logits, y, loss_mask):
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss_mask = loss_mask.view(-1)
        loss = F.cross_entropy(logits, y, reduction='none', ignore_index=-1)
        masked_loss = loss * loss_mask
        avg_loss = masked_loss.sum() / (loss_mask.sum() + 1e-9)
        return avg_loss

    def _collect_hook_metrics(self) -> dict:
        metrics = {}
        if self.hooks and is_main_process():
            for hook in self.hooks:
                metrics.update(hook.get_metric())
        return metrics

    def train_epoch(self, epoch, global_step):
        self.model.train()
        total_loss = 0
        epoch_start_time = time.perf_counter()

        if self.train_loader and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        pbar_total = len(self.train_loader) if self.train_loader else 0
        pbar = tqdm(self.train_loader, total=pbar_total, desc=f"Epoch {epoch} [Training]",
                    disable=not is_main_process())

        for i, (x, y, loss_mask) in enumerate(pbar):
            step_start_time = time.perf_counter()
            x, y, loss_mask = x.to(self.device), y.to(self.device), loss_mask.to(self.device)
            amp_context = torch.autocast(device_type=self.device,
                                         dtype=torch.bfloat16 if self.use_cpu_amp else torch.float16,
                                         enabled=(self.use_cpu_amp or self.use_gpu_amp))
            with amp_context:
                logits = self.model(x)
                loss = self._compute_loss(logits, y, loss_mask)
                loss_for_backward = loss / self.gradient_accumulation_steps
            loss_unscaled = loss.item()
            is_spike = False
            if self.last_train_loss is not None and loss_unscaled > (self.last_train_loss * self.loss_spike_threshold):
                is_spike = True;
                self.consecutive_spikes += 1
                if self.consecutive_spikes >= self.max_consecutive_spikes:
                    logging.error(f"连续Loss Spike次数达到{self.max_consecutive_spikes}，训练终止。")
                    raise RuntimeError("Training stopped due to excessive loss spikes.")
            else:
                self.consecutive_spikes = 0

            if self.last_train_loss is None:
                self.last_train_loss = loss_unscaled
            else:
                self.last_train_loss = 0.9 * self.last_train_loss + 0.1 * loss_unscaled

            if self.use_gpu_amp:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                if is_spike:
                    self.optimizer.zero_grad(set_to_none=True)
                    if is_main_process(): pbar.set_postfix_str(f"loss={loss_unscaled:.4f}, SPIKE_SKIPPED!",
                                                               refresh=True)
                    continue
                if self.use_gpu_amp: self.scaler.unscale_(self.optimizer)

                model_for_clip = self.model.module if get_world_size() > 1 else self.model
                if len(self.grad_norm_history) < self.grad_norm_history.maxlen:
                    current_clip_value = self.static_clip_grad_norm
                else:
                    percentile_val = torch.tensor(list(self.grad_norm_history)).quantile(
                        self.grad_clip_percentile).item()
                    current_clip_value = percentile_val * self.dynamic_clip_factor
                global_grad_norm = torch.nn.utils.clip_grad_norm_(model_for_clip.parameters(), current_clip_value)

                if not torch.isnan(global_grad_norm) and not torch.isinf(
                    global_grad_norm): self.grad_norm_history.append(global_grad_norm.item())

                if self.use_gpu_amp:
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step();
                self.optimizer.zero_grad(set_to_none=True);
                global_step += 1

                if is_main_process():
                    step_duration_ms = (time.perf_counter() - step_start_time) * 1000
                    mem_gb = self.process.memory_info().rss / (1024 ** 3) if self.process else -1
                    pbar.set_postfix({"loss": f"{loss_unscaled:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                                      "gnorm": f"{global_grad_norm.item():.3f}", "ms": f"{step_duration_ms:.2f}"},
                                     refresh=False)
                    if global_step % self.log_interval == 0:
                        metrics = {'train/loss_step': loss_unscaled, 'lr': self.scheduler.get_last_lr()[0],
                                   'gradients/global_norm': global_grad_norm.item(),
                                   'gradients/dynamic_clip_value': current_clip_value,
                                   'perf/step_duration_ms': step_duration_ms, 'perf/memory_rss_gb': mem_gb,
                                   **self._collect_hook_metrics()}
                        self.logger.log(metrics, step=global_step)
                    if global_step % self.save_interval == 0: self.ckpt_manager.save(epoch, -1, is_best=False)
            total_loss += loss_unscaled
        if is_main_process(): pbar.refresh()

        return total_loss / pbar_total if pbar_total > 0 else 0, global_step, time.perf_counter() - epoch_start_time

    @torch.no_grad()
    def evaluate(self, epoch, global_step):
        # [CRITICAL DDP FIX]
        # 所有进程都需要进入这个函数并迭代dataloader，以确保分布式采样器同步。
        # 否则，非主进程会跳过验证并抢先进入下一个训练epoch，导致主进程在dataloader上死锁。
        if self.val_loader is None:
            # 即使没有验证集，也要确保所有进程行为一致
            if get_world_size() > 1:
                torch.distributed.barrier()
            return {'loss': -1, 'perplexity': -1}

        self.model.eval()
        total_loss = 0

        # 所有进程都创建pbar，但只有主进程会显示它
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Validation]", disable=not is_main_process())

        for x, y, loss_mask in pbar:
            x, y, loss_mask = x.to(self.device), y.to(self.device), loss_mask.to(self.device)

            # 只有主进程进行实际的计算和指标累加
            if is_main_process():
                amp_context = torch.autocast(device_type=self.device,
                                             dtype=torch.bfloat16 if self.use_cpu_amp else torch.float16,
                                             enabled=(self.use_cpu_amp or self.use_gpu_amp))
                with amp_context:
                    logits = self.model(x)
                    loss = self._compute_loss(logits, y, loss_mask)
                total_loss += loss.item()
                pbar.set_postfix_str(f"loss={loss.item():.4f}")

        # 计算最终指标
        avg_loss = total_loss / len(self.val_loader) if is_main_process() and len(self.val_loader) > 0 else -1

        # 在DDP模式下，我们需要将主进程计算出的avg_loss广播给所有其他进程，
        # 尽管它们不会使用它，但这是一种好的实践，可以用于未来的同步需求。
        if get_world_size() > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            torch.distributed.broadcast(loss_tensor, src=0)
            avg_loss = loss_tensor.item()

        return {'loss': avg_loss, 'perplexity': calculate_perplexity(avg_loss)}

    def run(self, max_epochs: int, start_epoch: int):
        if is_main_process(): print("\n--- 6. 开始训练 ---")
        global_step = start_epoch * (len(self.train_loader) if self.train_loader else 0)
        for epoch in range(start_epoch, max_epochs):
            avg_train_loss, global_step, epoch_duration_s = self.train_epoch(epoch, global_step)
            eval_metrics = self.evaluate(epoch, global_step)

            # [CRITICAL DDP FIX] 这个屏障现在变得至关重要。
            # 它确保在主进程完成日志记录和保存检查点之前，没有任何进程可以进入下一个循环。
            if get_world_size() > 1:
                torch.distributed.barrier()

            if is_main_process():
                summary_metrics = {'train/loss_epoch': avg_train_loss, 'epoch_duration_s': epoch_duration_s,
                                   **{'val/' + k: v for k, v in eval_metrics.items()}}
                self.logger.log(summary_metrics, step=epoch)
                is_best = eval_metrics['loss'] != -1 and eval_metrics['loss'] < self.ckpt_manager.best_val_loss
                if is_best: self.ckpt_manager.best_val_loss = eval_metrics['loss']
                self.ckpt_manager.save(epoch, eval_metrics['loss'], is_best)
        if is_main_process(): self.logger.finish()


# END OF FILE: pretrain/components/training_loop.py