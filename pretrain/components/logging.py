# FILE: pretrain/components/logging.py
"""
【可插拔日志系统】
【修复版】使用更精确的浮点数格式化。
"""
import logging
from pathlib import Path
from typing import List, Dict, Any

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
    def __init__(self, config: Dict[str, Any] = None):
        pass

    def log(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError

    def finish(self):
        pass


# --- Backend Implementations ---
class FileLogger(BaseLogger):
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "train.log"
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info("FileLogger initialized.")

    def log(self, metrics: Dict[str, Any], step: int):
        # --- 核心修复：使用 .6g 格式化，能自动处理小数值 ---
        # .6g 表示总共保留6位有效数字，并会自动选择最佳表示法（定点或科学计数）
        log_parts = [f"{k}={v:.6g}" for k, v in metrics.items()]
        log_str = f"Step/Epoch {step}: " + ", ".join(log_parts)
        logging.info(log_str)


class WandbLogger(BaseLogger):
    def __init__(self, project: str, run_name: str, log_dir: Path, config: Dict[str, Any] = None):
        if not wandb:
            raise ImportError("wandb is not installed. Please run 'pip install wandb'.")
        try:
            wandb.init(project=project, name=run_name, dir=log_dir, config=config)
            logging.info(f"WandbLogger initialized for project '{project}'.")
        except Exception as e:
            logging.error(f"Failed to initialize wandb: {e}")
            raise

    def log(self, metrics: Dict[str, Any], step: int):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
        logging.info("WandbLogger finished.")


class SwanlabLogger(BaseLogger):
    def __init__(self, project: str, experiment_name: str, log_dir: Path, config: Dict[str, Any] = None):
        if not swanlab:
            raise ImportError("swanlab is not installed. Please run 'pip install swanlab'.")
        try:
            swanlab.init(project=project, experiment_name=experiment_name, logdir=str(log_dir), config=config)
            logging.info(f"SwanlabLogger initialized for project '{project}'.")
        except Exception as e:
            logging.error(f"Failed to initialize swanlab: {e}")
            raise

    def log(self, metrics: Dict[str, Any], step: int):
        swanlab.log(metrics, step=step)

    def finish(self):
        pass


# --- Main Logger Dispatcher ---
class Logger:
    def __init__(self, loggers: List[BaseLogger]):
        self.loggers = loggers
        logging.info(f"Main Logger initialized with {len(self.loggers)} backend(s).")

    def log(self, metrics: Dict[str, Any], step: int):
        for logger in self.loggers:
            try:
                logger.log(metrics, step)
            except Exception as e:
                logging.error(f"Failed to log with {type(logger).__name__}: {e}")

    def finish(self):
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as e:
                logging.error(f"Failed to finish logger {type(logger).__name__}: {e}")
        logging.info("All loggers finished.")

# END OF FILE: pretrain/components/logging.py