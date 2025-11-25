# FILE: pretrain/train.py
# -*- coding: utf-8 -*-
"""
【v4.3 - 鲁棒性巅峰版】统一预训练/继续预训练脚本 (DDP + Compile + Auto-Fallback)
- [自举] Windows UTF-8 编码自动修复。
- [预检] 自动检测 C++ 编译器。如果没有安装 VS Build Tools，自动关闭编译以免崩溃。
- [兼容] MoE 架构自动适配 DDP 参数。
"""
import os
import sys
import subprocess
import shutil

# --- [Windows 兼容性补丁: 必须在任何逻辑执行前运行] ---
if os.name == 'nt' and os.environ.get('PYTHONUTF8') != '1':
    print("🔄 [系统自举] Windows 环境检测: 正在设置 PYTHONUTF8=1 并重启训练进程...")
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    try:
        ret = subprocess.call([sys.executable] + sys.argv, env=env)
        sys.exit(ret)
    except Exception as e:
        print(f"❌ 自举失败: {e}")
        sys.exit(1)
# -----------------------------------------------------

import torch
import argparse
from pathlib import Path
import time
import shutil

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ddp_utils import setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process

from utils.config_loader import load_config
from utils.builders import build_model, build_optimizer, build_scheduler, build_loggers
from pretrain.data_loader import get_pretrain_loaders
from pretrain.components.checkpointing import CheckpointManager
from pretrain.components.training_loop import Trainer
from pretrain.components.hooks import register_hooks

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None


def check_cxx_compiler() -> bool:
    """
    检查系统中是否存在 C++ 编译器 (cl.exe for Windows, g++ for Linux/others)。
    torch.compile(backend='inductor') 强依赖于 C++ 编译器。
    """
    if os.name == 'nt':
        # Windows 需要 Visual Studio Build Tools (cl.exe)
        # 或者 MinGW (g++)，但 inductor 对 MSVC 支持最好
        if shutil.which('cl') is not None:
            return True
        if shutil.which('g++') is not None:
            return True
        return False
    else:
        # Linux/Mac 通常预装 g++ 或 clang
        return shutil.which('c++') is not None or shutil.which('g++') is not None or shutil.which('clang++') is not None


def main():
    parser = argparse.ArgumentParser(description="[v4.3] 统一预训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="指向配置YAML文件的路径")
    parser.add_argument("--fast_dev_run", action="store_true", help="启用快速开发运行模式")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile (PyTorch 2.0+) 加速")
    args = parser.parse_args()

    setup_ddp()
    world_size = get_world_size()

    # --- 0. 配置与日志 ---
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    output_dir = None
    run_name = ""
    if is_main_process():
        base_output_dir = Path(cfg.output_dir)
        if args.fast_dev_run:
            run_name = "fast-dev-run"
            output_dir = base_output_dir / "pretrain" / run_name
            if output_dir.exists():
                print(f"🧹 fast_dev_run 模式 (主进程): 正在清理旧的开发目录 {output_dir}")
                shutil.rmtree(output_dir)
        else:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            run_name = cfg.run_name.format(timestamp=timestamp)
            output_dir = base_output_dir / "pretrain" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_loggers(cfg, output_dir, run_name)
    if is_main_process():
        print(f"配置加载自: {args.config_path}")
        print(f"所有输出将保存到: {output_dir}")

    # --- 1. 模型 ---
    cfg.model.use_activation_checkpointing = getattr(cfg.training, 'use_activation_checkpointing', False)
    model = build_model(cfg.model).to(cfg.device)

    # [性能优化] torch.compile 智能处理
    if args.compile:
        can_compile = True
        # 1. 检查编译器环境
        if not check_cxx_compiler():
            if is_main_process():
                print("\n⚠️  [警告] 未检测到 C++ 编译器 (cl.exe 或 g++)！")
                print("   torch.compile 需要 C++ 环境才能工作。")
                print("   -> Windows 用户请安装: 'Visual Studio Build Tools' (勾选 C++ 桌面开发)。")
                print("   -> 正在自动降级回 Eager 模式 (无编译) 继续运行...\n")
            can_compile = False

        # 2. 执行编译
        if can_compile:
            if is_main_process():
                print("🚀 正在编译模型 (torch.compile)... 首次迭代可能会变慢。")
            try:
                # Windows 下 inductor 偶尔会有路径问题，加个保护
                model = torch.compile(model, backend="inductor")
            except Exception as e:
                if is_main_process():
                    print(f"❌ 编译失败: {e}")
                    print("   -> 回退到 Eager 模式运行。")

    if world_size > 1:
        has_moe = cfg.model.num_experts > 1
        find_unused = has_moe

        if is_main_process() and has_moe:
            print("⚠️ 检测到 MoE 架构，已启用 DDP(find_unused_parameters=True)。")

        model = DDP(
            model,
            device_ids=None if cfg.device == 'cpu' else [int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=find_unused
        )
        if is_main_process():
            print(f"模型已用 DDP 包装 (Rank {get_rank()})。")

    # --- 2. 数据、优化器、调度器、混合精度 ---
    train_limit = getattr(cfg.data, 'train_data_limit', None)
    val_limit = getattr(cfg.data, 'val_data_limit', None)

    train_loader, val_loader = get_pretrain_loaders(
        tokenizer_name=cfg.data.tokenizer_name, data_dir=Path(cfg.data.data_dir),
        block_size=cfg.model.max_seq_len, batch_size=cfg.training.batch_size,
        train_data_limit=train_limit, val_data_limit=val_limit,
        ddp_rank=get_rank(), ddp_world_size=world_size
    )

    model_for_optimizer = model.module if world_size > 1 else model
    optimizer = build_optimizer(model_for_optimizer, cfg.training)
    max_iters = len(train_loader) * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg.training, max_iters)
    scaler = GradScaler() if cfg.device == 'cuda' and GradScaler else None

    # --- 3. 检查点管理器与加载 ---
    if is_main_process():
        print("\n--- 4. 初始化检查点管理器 (仅主进程) ---")

    ckpt_dir = output_dir / "checkpoints" if is_main_process() else None
    ckpt_manager = CheckpointManager(ckpt_dir, model_for_optimizer, optimizer, scheduler, scaler)
    start_epoch = 0

    load_ckpt_path = getattr(cfg.training, 'load_from_checkpoint', "none")
    load_only_model = getattr(cfg.training, 'load_only_model', False)

    if load_ckpt_path != "none":
        if is_main_process():
            print(f"检测到加载请求: {load_ckpt_path}")
        start_epoch = ckpt_manager.load(load_ckpt_path, load_only_model=load_only_model)

    if world_size > 1:
        torch.distributed.barrier()

    # --- 4. 钩子与训练器 ---
    if is_main_process():
        print("--- 1.1. 为模型注册监控钩子 ---")
        try:
            # 注意：编译后的模型注册 hook 可能会受限，这里尽力而为
            hooks = register_hooks(model_for_optimizer)
            print(f"✅ 已成功注册 {len(hooks)} 个钩子用于监控内部状态。")
        except Exception as e:
            print(f"⚠️ 钩子注册失败 (可能受 torch.compile 影响): {e}")
            hooks = None
    else:
        hooks = None

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=cfg.device,
        logger=logger, ckpt_manager=ckpt_manager,
        hooks=hooks,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_interval=getattr(cfg.logging, 'log_interval', 10),
        save_interval=getattr(cfg.checkpointing, 'save_interval', 1000),
        scaler=scaler,
        clip_grad_norm=getattr(cfg.training, 'clip_grad_norm', 1.0),
        loss_spike_threshold=getattr(cfg.training, 'loss_spike_threshold', 5.0),
        max_consecutive_spikes=getattr(cfg.training, 'max_consecutive_spikes', 5),
        grad_norm_history_size=getattr(cfg.training, 'grad_norm_history_size', 100),
        grad_clip_percentile=getattr(cfg.training, 'grad_clip_percentile', 0.9),
        dynamic_clip_factor=getattr(cfg.training, 'dynamic_clip_factor', 1.5)
    )
    trainer.run(cfg.training.max_epochs, start_epoch)

    if world_size > 1:
        torch.distributed.barrier()

    # --- DDP 清理 ---
    cleanup_ddp()


if __name__ == "__main__":
    main()
# END OF FILE: pretrain/train.py