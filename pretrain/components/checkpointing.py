# FILE: pretrain/components/checkpointing.py
"""
【v2.0 - 增加日志细节】实现检查点管理。
"""
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Optional

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
        # [新增] 打印初始化信息
        print(f"✅ 检查点管理器已初始化。检查点将保存到: '{self.checkpoint_dir}'")

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
        # 使用 logging 模块，信息会进入文件和 SwanLab 控制台
        logging.info(f"Saved latest checkpoint to {latest_ckpt_path}")

        if is_best:
            best_ckpt_path = self.checkpoint_dir / "ckpt_best.pth"
            torch.save(state, best_ckpt_path)
            logging.info(f"Saved best checkpoint to {best_ckpt_path} (New best Val Loss: {val_loss:.4f})")

    def load(self, resume_from: str = "latest") -> int:
        ckpt_path_map = {
            "latest": self.checkpoint_dir / "ckpt_latest.pth",
            "best": self.checkpoint_dir / "ckpt_best.pth"
        }
        ckpt_path = ckpt_path_map.get(resume_from)

        if not ckpt_path or not ckpt_path.exists():
            print(f"⚠️ 检查点 '{resume_from}' 未找到 at '{ckpt_path}'. 将从头开始训练。")
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

            print(f"✅ 成功从 '{ckpt_path}' 加载检查点。将从 epoch {start_epoch} 继续。")
            return start_epoch
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}。将从头开始训练。")
            return 0
# END OF FILE: pretrain/components/checkpointing.py