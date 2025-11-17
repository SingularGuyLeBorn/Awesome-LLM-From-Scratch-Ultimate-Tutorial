# FILE: utils/ddp_utils.py
# -*- coding: utf-8 -*-
"""
[新增] 分布式数据并行 (DDP) 辅助工具。
"""
import os
import torch
import torch.distributed as dist


def is_ddp_enabled() -> bool:
    """检查DDP环境变量是否已设置。"""
    return "WORLD_SIZE" in os.environ and "RANK" in os.environ


def setup_ddp():
    """
    初始化分布式进程组。
    """
    if not is_ddp_enabled():
        return

    # WORLD_SIZE: 全局总进程数
    # RANK: 当前进程的全局唯一ID (0 to WORLD_SIZE - 1)
    # LOCAL_RANK: 当前节点上的进程唯一ID
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # PyTorch推荐使用 'nccl' 后端进行GPU训练，'gloo' 用于CPU
    # 由于我们是CPU友好项目，默认使用 'gloo'
    backend = 'gloo'

    # 如果有CUDA设备，则切换为 'nccl'
    if torch.cuda.is_available():
        backend = 'nccl'
        torch.cuda.set_device(local_rank)

    print(f"Initializing DDP for rank {rank}/{world_size} (local_rank: {local_rank}) with backend '{backend}'")
    dist.init_process_group(backend=backend)


def cleanup_ddp():
    """
    清理分布式进程组。
    """
    if is_ddp_enabled():
        dist.destroy_process_group()
        print("DDP process group destroyed.")


def get_rank() -> int:
    """获取当前进程的全局排名。如果未启用DDP，则返回0。"""
    return dist.get_rank() if is_ddp_enabled() else 0


def get_world_size() -> int:
    """获取总进程数。如果未启用DDP，则返回1。"""
    return dist.get_world_size() if is_ddp_enabled() else 1


def is_main_process() -> bool:
    """检查当前是否为主进程 (rank 0)。"""
    return get_rank() == 0