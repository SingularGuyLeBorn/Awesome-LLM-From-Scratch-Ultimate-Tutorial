# FILE: pretrain/components/logging.py
"""
【v3.1 - 最终净化版】
- ConsoleLogger 的职责被最终明确：只负责在Epoch结束时打印摘要。
- 所有 step 级的控制台输出完全交由 tqdm 处理。
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
import os
import datetime

# --- 动态导入 ---
try:
    import wandb
except ImportError:
    wandb = None
try:
    import swanlab
except ImportError:
    swanlab = None


# --- Base Logger Class ---
class BaseLogger:
    def log(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError

    def finish(self):
        pass


# --- Backend Implementations ---
class FileLogger(BaseLogger):
    """记录所有详细指标到 train.log，用于机器分析和深度调试。"""

    def __init__(self, log_dir: Path, **kwargs):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "train.log"
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file)]
        )
        initial_messages = [
            "--- 训练环境诊断 ---",
            f"PyTorch Version: {torch.__version__}",
            f"CUDA Available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            initial_messages.append(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        initial_messages.append(f"Number of CPUs: {os.cpu_count()}")
        initial_messages.append("--- 诊断结束 ---")
        initial_messages.append(f"FileLogger initialized. Detailed machine logs: {log_file}")
        for msg in initial_messages:
            logging.info(msg)

    def log(self, metrics: Dict[str, Any], step: int):
        full_log_parts = [f"{k}={v:.6g}" for k, v in metrics.items()]
        full_log_str = f"Step/Epoch {step}: " + ", ".join(full_log_parts)
        logging.info(full_log_str)


class HumanReadableLogger(BaseLogger):
    """创建一个简洁、对齐的 summary.log，方便人类快速回顾。"""

    def __init__(self, log_dir: Path, **kwargs):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "summary.log"
        header = f"{'Epoch':<10} | {'Train Loss':<15} | {'Val Loss':<15} | {'Val Perplexity':<20} | {'Duration (s)':<15}\n"
        separator = f"{'-' * 10} | {'-' * 15} | {'-' * 15} | {'-' * 20} | {'-' * 15}\n"
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(separator)

    def log(self, metrics: Dict[str, Any], step: int):
        if 'val/loss' in metrics and 'train/loss_epoch' in metrics:
            epoch = step
            train_loss = metrics.get('train/loss_epoch', float('nan'))
            val_loss = metrics.get('val/loss', float('nan'))
            val_ppl = metrics.get('val/perplexity', float('nan'))
            duration = metrics.get('epoch_duration_s', -1)
            log_line = (f"{epoch:<10d} | {train_loss:<15.4f} | "
                        f"{val_loss:<15.4f} | {val_ppl:<20.2f} | {duration:<15.2f}\n")
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)

    def finish(self):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Training finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")


class ConsoleLogger(BaseLogger):
    """[最终版] 专职的控制台日志记录器, 只负责Epoch总结。"""

    def __init__(self, verbose: bool = False, **kwargs):
        # verbose 参数被保留，以备将来为Epoch总结增加详细模式
        self.verbose = verbose
        # 打印初始环境信息
        initial_messages = [
            "--- 训练环境诊断 ---",
            f"PyTorch Version: {torch.__version__}",
            f"CUDA Available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            initial_messages.append(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        initial_messages.append(f"Number of CPUs: {os.cpu_count()}")
        initial_messages.append("--- 诊断结束 ---")
        for msg in initial_messages:
            print(msg)

    def log(self, metrics: Dict[str, Any], step: int):
        # [核心修复] 只在Epoch结束时打印总结，完全忽略step级日志
        if 'val/loss' in metrics:
            train_loss = metrics.get('train/loss_epoch', -1)
            val_loss = metrics.get('val/loss', -1)
            val_ppl = metrics.get('val/perplexity', -1)
            duration = metrics.get('epoch_duration_s', -1)
            print(
                f"\nEpoch {step} Summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}, Duration={duration:.2f}s")


class WandbLogger(BaseLogger):
    def __init__(self, project: str, run_name: str, log_dir: Path, **kwargs):
        if not wandb: raise ImportError("wandb not installed.")
        wandb.init(project=project, name=run_name, dir=log_dir, config=kwargs.get('config'))
        print(f"WandbLogger initialized for project '{project}'.")

    def log(self, metrics: Dict[str, Any], step: int): wandb.log(metrics, step=step)

    def finish(self): wandb.finish(); print("WandbLogger finished.")


class SwanlabLogger(BaseLogger):
    def __init__(self, project: str, experiment_name: str, log_dir: Path, **kwargs):
        if not swanlab: raise ImportError("swanlab not installed.")
        swanlab.init(project=project, experiment_name=experiment_name, logdir=str(log_dir), config=kwargs.get('config'))
        print(f"SwanlabLogger initialized for project '{project}'.")

    def log(self, metrics: Dict[str, Any], step: int): swanlab.log(metrics, step=step)

    def finish(self): pass


# --- Main Logger Dispatcher ---
class Logger:
    def __init__(self, loggers: List[BaseLogger]):
        self.loggers = loggers
        print(f"Main Logger initialized with {len(self.loggers)} backend(s).")

    def log(self, metrics: Dict[str, Any], step: int):
        for logger in self.loggers:
            try:
                logger.log(metrics, step)
            except Exception as e:
                print(f"ERROR: Failed to log with {type(logger).__name__}: {e}")

    def finish(self):
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as e:
                print(f"ERROR: Failed to finish logger {type(logger).__name__}: {e}")
        print("All loggers finished.")
# END OF FILE: pretrain/components/logging.py