# FILE: utils/builders.py
"""
[v3.2 - Builder Complete]
- 提供完整的构建函数集合。
- 链接 pretrain.components.optimizers.get_optimizer。
- 包含 QLoRA 加载、参数分析、模型/优化器/调度器/日志器构建。
"""
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace

from .ddp_utils import is_main_process, is_ddp_enabled

from models.transformer import Transformer, ModelArgs
from models.value_model import ValueModel
from models.reward_model import RewardModel
from models.blocks.feedforward.moe import MoELayer

# 引入 QLoRA 相关组件
from finetune.peft.qlora.linear4bit import Linear4bit
from finetune.peft.qlora.qlora import replace_linear_with_qlora, QLoRALayer

# [核心] 导入统一的优化器工厂
from pretrain.components.optimizers import get_optimizer
from pretrain.components.scheduler import get_lr_scheduler
from pretrain.components.logging import (
    Logger, ConsoleLogger, FileLogger, HumanReadableLogger,
    WandbLogger, SwanlabLogger
)


def analyze_model_parameters(model: nn.Module):
    total_params = 0
    active_params = 0
    emb_params = sum(p.numel() for p in model.tok_embeddings.parameters())
    head_params = 0
    if not (model.tok_embeddings.weight is model.lm_head.weight):
        head_params = sum(p.numel() for p in model.lm_head.parameters())
    norm_params = sum(p.numel() for p in model.norm.parameters())
    shared_dense_params = emb_params + head_params + norm_params
    attn_params_total = 0
    ffn_params_total = 0
    ffn_params_active = 0
    moe_info = {"layers": 0, "experts": 0, "topk": 0}
    sample_expert_size = 0
    sample_router_size = 0
    for layer in model.layers:
        blk_attn_params = sum(p.numel() for p in layer.attention.parameters())
        blk_norm_params = sum(p.numel() for p in layer.attention_norm.parameters()) + \
                          sum(p.numel() for p in layer.ffn_norm.parameters())
        curr_layer_fixed = blk_attn_params + blk_norm_params
        attn_params_total += curr_layer_fixed
        if isinstance(layer.feed_forward, MoELayer):
            moe_info["layers"] += 1
            moe_info["experts"] = layer.feed_forward.num_experts
            moe_info["topk"] = layer.feed_forward.num_experts_per_tok
            r_p = sum(p.numel() for p in layer.feed_forward.router.parameters())
            sample_router_size = r_p
            one_expert = sum(p.numel() for p in layer.feed_forward.experts[0].parameters())
            sample_expert_size = one_expert
            layer_ffn_total = r_p + (moe_info["experts"] * one_expert)
            layer_ffn_active = r_p + (moe_info["topk"] * one_expert)
            ffn_params_total += layer_ffn_total
            ffn_params_active += layer_ffn_active
        else:
            p = sum(p.numel() for p in layer.feed_forward.parameters())
            ffn_params_total += p
            ffn_params_active += p
    total_params = shared_dense_params + attn_params_total + ffn_params_total
    active_params = shared_dense_params + attn_params_total + ffn_params_active
    print("=" * 60)
    print(f"{'Model Architecture & Parameter Deep Dive':^60}")
    print("=" * 60)
    print(f" [Summary]")
    print(f" ▶ Total Parameters  : {total_params / 1e6:6.2f} M  (Disk Size)")
    print(f" ▶ Active Parameters : {active_params / 1e6:6.2f} M  (Compute Cost)")
    print(f" ▶ Sparsity Ratio    : {active_params / total_params:6.2%}   (Lower is more sparse)")
    print("-" * 60)
    print(f" [Component Breakdown (Total / Active)]")
    print(f" 1. Embeddings & Head : {shared_dense_params / 1e6:.2f} M (Always Active)")
    print(f" 2. Attention Layers  : {attn_params_total / 1e6:.2f} M (Always Active)")
    if moe_info["layers"] > 0:
        print(f" 3. MoE FFN Layers    : {ffn_params_total / 1e6:.2f} M -> {ffn_params_active / 1e6:.2f} M (Sparse!)")
        print("-" * 60)
        print(f" [MoE Detail Analysis]")
        print(
            f"   - Configuration    : {moe_info['layers']} MoE Layers | {moe_info['experts']} Experts Total | Top-{moe_info['topk']} Active")
        print(f"   - Single Expert    : {sample_expert_size / 1e6:.4f} M params")
        print(f"   - Router (Gate)    : {sample_router_size} params")
        print(f"   - Logic Check      : Each token uses {moe_info['topk']} experts out of {moe_info['experts']}")
    else:
        print(f" 3. Dense FFN Layers  : {ffn_params_total / 1e6:.2f} M (Dense)")
    print("=" * 60)
    print("\n")
    return total_params, active_params


def build_model(model_config: SimpleNamespace) -> nn.Module:
    if is_main_process():
        print("\n--- 1. 初始化模型 ---")
    model_args = ModelArgs(**vars(model_config))
    model = Transformer(model_args)
    if is_main_process():
        analyze_model_parameters(model)
    return model


def build_value_model(model_config: SimpleNamespace) -> nn.Module:
    if is_main_process():
        print("--- 1.1. 初始化价值模型 (Critic) ---")
    model_args = ModelArgs(**vars(model_config))
    model = ValueModel(model_args)
    if is_main_process():
        print(f"价值模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model


def build_reward_model(model_config: SimpleNamespace) -> nn.Module:
    if is_main_process():
        print("--- 1.2. 初始化奖励模型 ---")
    model_args = ModelArgs(**vars(model_config))
    model = RewardModel(model_args)
    if is_main_process():
        print(f"奖励模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model


def load_qlora_model_for_inference(config: SimpleNamespace, base_ckpt_path: str, adapter_ckpt_path: str,
                                   device: str = "cpu") -> nn.Module:
    print(f"\n--- QLoRA Inference Setup ---")
    print(f"1. Building Base Model configuration...")
    model = build_model(config.model)

    print(f"2. Loading Base Checkpoint: {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
    model.load_state_dict(base_ckpt['model_state_dict'], strict=False)

    qlora_cfg = getattr(config, 'qlora', SimpleNamespace())
    r = getattr(qlora_cfg, 'r', 8)
    alpha = getattr(qlora_cfg, 'alpha', 16)
    dropout = getattr(qlora_cfg, 'dropout', 0.05)
    target_modules = getattr(qlora_cfg, 'target_modules', ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"])

    print(f"3. Quantizing to 4-bit NF4 (Target Modules: {target_modules})...")
    compute_dtype = torch.float32

    replace_linear_with_qlora(
        model,
        rank=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        compute_dtype=compute_dtype
    )

    print(f"4. Loading Adapter Weights: {adapter_ckpt_path}")
    adapter_ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    if 'model_state_dict' in adapter_ckpt:
        state_dict = adapter_ckpt['model_state_dict']
    else:
        state_dict = adapter_ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ QLoRA Model Ready on {device}.")
    return model


def build_optimizer(model: nn.Module, train_config: SimpleNamespace) -> torch.optim.Optimizer:
    if is_main_process():
        print("\n--- 3. 初始化优化器 ---")

    # 直接调用统一工厂
    optimizer = get_optimizer(model, train_config)

    if is_main_process():
        print(f"✅ 优化器构建完成 (类型: {type(optimizer).__name__})")
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, train_config: SimpleNamespace,
                    max_iters: int) -> torch.optim.lr_scheduler._LRScheduler:
    if is_main_process():
        print("--- 3.1. 初始化调度器 ---")
    warmup_iters = int(max_iters * getattr(train_config, 'warmup_ratio', 0))
    min_lr = train_config.learning_rate * getattr(train_config, 'min_lr_ratio', 0)
    scheduler = get_lr_scheduler(optimizer, warmup_iters, max_iters, min_lr)
    return scheduler


def build_loggers(config: SimpleNamespace, output_dir: Path, run_name: str) -> Logger:
    if not is_main_process():
        return Logger([])

    print("\n--- 0. 初始化日志系统 ---")
    loggers_to_use = []

    console_cfg = getattr(config, 'console', SimpleNamespace(verbose=False))
    loggers_to_use.append(ConsoleLogger(**vars(console_cfg)))
    loggers_to_use.append(FileLogger(log_dir=output_dir))
    loggers_to_use.append(HumanReadableLogger(log_dir=output_dir))

    logging_cfg = getattr(config, 'logging', SimpleNamespace())
    wandb_cfg = getattr(logging_cfg, 'wandb', SimpleNamespace(enable=False))
    if wandb_cfg and wandb_cfg.enable:
        try:
            loggers_to_use.append(WandbLogger(
                project=wandb_cfg.project,
                run_name=run_name,
                log_dir=output_dir,
                config=vars(config)
            ))
        except ImportError as e:
            print(f"WARNING: 无法添加WandbLogger: {e}.")

    swanlab_cfg = getattr(logging_cfg, 'swanlab', SimpleNamespace(enable=False))
    if swanlab_cfg and swanlab_cfg.enable:
        if is_ddp_enabled():
            print("⚠️ 警告: 检测到DDP环境。为保证稳定性，已自动禁用 Swanlab。")
        else:
            try:
                exp_name = getattr(swanlab_cfg, 'experiment_name', run_name).format(run_name=run_name)
                loggers_to_use.append(SwanlabLogger(
                    project=swanlab_cfg.project,
                    experiment_name=exp_name,
                    log_dir=output_dir,
                    config=vars(config)
                ))
            except ImportError as e:
                print(f"WARNING: 无法添加SwanlabLogger: {e}.")

    return Logger(loggers_to_use)

# END OF FILE: utils/builders.py