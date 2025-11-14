# FILE: pretrain/components/checkpointing.py
"""
【重构版】实现检查点管理，支持保存和加载训练状态，
并为GPU混合精度训练的GradScaler状态提供支持。
"""
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Optional

# 尝试导入GradScaler，如果失败则设为None
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                 scaler: Optional[GradScaler] = None):
        """
        Args:
            checkpoint_dir (Path): 保存检查点的目录。
            model (nn.Module): 要保存的模型。
            optimizer (torch.optim.Optimizer): 要保存的优化器。
            scheduler: 要保存的学习率调度器。
            scaler (GradScaler, optional): 混合精度训练的GradScaler。
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.best_val_loss = float('inf')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()

        latest_ckpt_path = self.checkpoint_dir / "ckpt_latest.pth"
        torch.save(state, latest_ckpt_path)
        logging.info(f"Saved latest checkpoint to {latest_ckpt_path}")

        if is_best:
            best_ckpt_path = self.checkpoint_dir / "ckpt_best.pth"
            torch.save(state, best_ckpt_path)
            logging.info(f"Saved best checkpoint to {best_ckpt_path} (Val Loss: {val_loss:.4f})")

    def load(self, resume_from: str = "latest") -> int:
        ckpt_path_map = {
            "latest": self.checkpoint_dir / "ckpt_latest.pth",
            "best": self.checkpoint_dir / "ckpt_best.pth"
        }
        ckpt_path = ckpt_path_map.get(resume_from)

        if not ckpt_path or not ckpt_path.exists():
            logging.warning(f"Checkpoint file for '{resume_from}' not found at {ckpt_path}. Starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            logging.info(f"Successfully loaded checkpoint from {ckpt_path}. Resuming from epoch {start_epoch}.")
            return start_epoch
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {ckpt_path}: {e}")
            logging.warning("Starting from scratch.")
            return 0


# END OF FILE: pretrain/components/checkpointing.py