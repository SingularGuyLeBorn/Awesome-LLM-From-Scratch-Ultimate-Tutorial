# FILE: pretrain/components/checkpointing.py
"""
【v2.3 - DDP 支持】实现检查点管理。
- save 方法现在只在主进程执行。
"""
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Optional

# [核心新增]
from utils.ddp_utils import is_main_process

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


class CheckpointManager:
    def __init__(self, checkpoint_dir: Optional[Path], model: nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                 scaler: Optional[GradScaler] = None):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.best_val_loss = float('inf')
        if is_main_process() and self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ 检查点管理器已初始化。检查点将保存到: '{self.checkpoint_dir}'")

    def save(self, epoch: int, val_loss: float, is_best: bool = False):
        # [核心修改] 只有主进程才执行保存操作
        if not is_main_process() or not self.checkpoint_dir:
            return

        # 在DDP模式下，我们需要保存 unwrapped model 的 state_dict
        model_state_dict = self.model.module.state_dict() if isinstance(self.model,
                                                                        nn.parallel.DistributedDataParallel) else self.model.state_dict()

        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': model_state_dict,
        }
        if self.optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()

        latest_ckpt_path = self.checkpoint_dir / "ckpt_latest.pth"
        torch.save(state, latest_ckpt_path)
        logging.info(f"Saved latest checkpoint to {latest_ckpt_path}")

        if is_best:
            best_ckpt_path = self.checkpoint_dir / "ckpt_best.pth"
            torch.save(state, best_ckpt_path)
            logging.info(f"Saved best checkpoint to {best_ckpt_path} (New best Val Loss: {val_loss:.4f})")

    def load(self, resume_from: str = "latest", load_only_model: bool = False) -> int:
        # 在DDP中，所有进程都需要加载模型权重，所以这个函数不需要 `is_main_process` 守护
        if not self.checkpoint_dir and resume_from not in ["latest", "best"]:
            ckpt_path = Path(resume_from)
        elif not self.checkpoint_dir:
            print("⚠️ 在非主进程中无法解析 'latest' 或 'best' 检查点路径。将从头开始。")
            return 0
        else:
            ckpt_path_map = {
                "latest": self.checkpoint_dir / "ckpt_latest.pth",
                "best": self.checkpoint_dir / "ckpt_best.pth"
            }
            ckpt_path = ckpt_path_map.get(resume_from, Path(resume_from))

        if not ckpt_path.exists():
            if is_main_process():
                print(f"⚠️ 检查点 '{resume_from}' 未找到 at '{ckpt_path}'. 将从头开始训练。")
            return 0

        try:
            # 所有进程都从CPU加载，以节省GPU内存
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            # 处理DDP和非DDP模型的状态字典键名差异
            model_to_load = self.model.module if isinstance(self.model,
                                                            nn.parallel.DistributedDataParallel) else self.model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])

            if is_main_process():
                print(f"✅ 成功从 '{ckpt_path}' 加载了模型权重。")

            if load_only_model:
                if is_main_process():
                    print("   -> 模式: 仅加载模型权重。优化器和调度器将重新初始化。")
                return 0

            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            if is_main_process():
                print(f"✅ 成功恢复完整训练状态。将从 epoch {start_epoch} 继续。")
            return start_epoch
        except Exception as e:
            if is_main_process():
                print(f"❌ 加载检查点失败: {e}。将从头开始训练。")
            return 0
# END OF FILE: pretrain/components/checkpointing.py