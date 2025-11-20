# FILE: pretrain/components/optimizers.py
"""
【优化器全家桶 v3.3 - Stability Hotfix】
- 修复 Muon μ_err 过大的问题。
- 在 Newton-Schulz 迭代中强制使用 float32 精度，确保 CPU 上的数值稳定性。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from typing import Any, Dict, List

from utils.ddp_utils import is_main_process


# ==============================================================================
# Core: Newton-Schulz Iteration (High Precision)
# ==============================================================================

def zeroth_power_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> tuple[torch.Tensor, float]:
    """
    使用 Newton-Schulz 迭代计算矩阵的近似正交化。

    [v3.3 修复]: 强制内部使用 float32 计算。
    原因: bfloat16 在 CPU 上进行多次矩阵乘法迭代时，精度不足导致无法收敛到正交矩阵(I)，
          从而导致 μ_err 很大 (3.6+)。
    """
    assert G.ndim >= 2

    # 展平
    if G.ndim > 2:
        G = G.view(G.size(0), -1)

    # 转置处理 (确保 Rows <= Cols)
    transposed = False
    if G.size(0) > G.size(1):
        transposed = True
        G = G.mT

    # [核心修复] 强制转为 float32 以保证迭代收敛
    original_dtype = G.dtype
    X = G.float()

    # 归一化光谱范数
    # 使用 Frobenius norm 近似
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # 官方推荐系数
    a, b, c = (3.4445, -4.7750, 2.0315)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # 计算误差 (使用 FP32 计算更准)
    with torch.no_grad():
        if X.size(0) <= 2048:
            I = torch.eye(X.size(0), device=X.device, dtype=torch.float32)
            err = (X @ X.mT - I).norm().item()
        else:
            err = 0.0

    if transposed:
        X = X.mT

    # 转回原始精度 (通常是 bf16)
    return X.to(original_dtype), err


# ==============================================================================
# Core: Update Functions
# ==============================================================================

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)

    update = momentum
    if nesterov:
        update = grad.lerp_(momentum, beta)

    update, err = zeroth_power_via_newtonschulz5(update, steps=ns_steps)

    # Shape Scaling
    if update.size(-1) > 1:
        scale = max(1, grad.size(-2) / grad.size(-1)) ** 0.5
        update.mul_(scale)

    return update, err


def adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])

    bias_correction1 = 1 - betas[0] ** step
    bias_correction2 = 1 - betas[1] ** step

    step_size = 1.0 / bias_correction1
    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

    return exp_avg * step_size / denom


# ==============================================================================
# Optimizer: Muon (Hybrid)
# ==============================================================================

class Muon(optim.Optimizer):
    def __init__(self, param_groups):
        defaults = dict(lr=0.02, weight_decay=0.01, momentum=0.95,
                        betas=(0.9, 0.95), eps=1e-8, use_muon=False)
        super().__init__(param_groups, defaults)
        self.metrics = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_ortho_err = 0.0
        muon_count = 0

        for group in self.param_groups:
            if group.get("use_muon", False):
                # Muon Group
                lr = group["lr"]
                wd = group["weight_decay"]
                momentum_beta = group["momentum"]
                ns_steps = group.get("ns_steps", 5)

                for p in group["params"]:
                    if p.grad is None: continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update, err = muon_update(p.grad, state["momentum_buffer"],
                                              beta=momentum_beta, ns_steps=ns_steps)

                    if err > 0:
                        total_ortho_err += err
                        muon_count += 1

                    if wd > 0:
                        p.mul_(1 - lr * wd)

                    p.add_(update.view_as(p), alpha=-lr)

            else:
                # AdamW Group
                lr = group["lr"]
                wd = group["weight_decay"]
                betas = group["betas"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None: continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], betas, eps)

                    if wd > 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        if muon_count > 0:
            self.metrics["ortho_err"] = total_ortho_err / muon_count

        return loss

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics


# ==============================================================================
# Other Optimizers
# ==============================================================================

class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss

    def get_metrics(self) -> Dict[str, float]:
        return {}


class SophiaG(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.metrics = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        sum_hessian = 0.0
        count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                lr = group['lr']
                beta1, beta2 = group['betas']
                rho = group['rho']
                weight_decay = group['weight_decay']

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                hessian = state['hessian']
                state['step'] += 1

                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if count < 10:
                    sum_hessian += hessian.mean().item()
                    count += 1

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                ratio = (exp_avg / (torch.sqrt(hessian) + 1e-15).add_(rho)).clamp(min=-1.0, max=1.0)
                p.add_(ratio, alpha=-lr)

        if count > 0:
            self.metrics["hessian_mean"] = sum_hessian / count

        return loss

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics


def get_optimizer(model: nn.Module, train_config: Any) -> optim.Optimizer:
    lr = train_config.learning_rate
    weight_decay = train_config.weight_decay
    opt_type = getattr(train_config, 'optimizer_type', 'adamw').lower()

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    if not param_dict:
        raise ValueError("No trainable parameters found.")

    if opt_type == 'muon':
        muon_params = []
        adam_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad: continue

            if p.ndim >= 2 and "embed" not in name and "head" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)

        muon_lr = 0.02
        adam_lr = lr

        param_groups = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr,
                "weight_decay": weight_decay,
                "momentum": 0.95,
                "ns_steps": 5
            },
            {
                "params": adam_params,
                "use_muon": False,
                "lr": adam_lr,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
                "eps": 1e-8
            }
        ]

        if is_main_process():
            print(f"✅ Muon Strategy Applied:")
            print(f"   - Muon Group: {len(muon_params)} tensors (LR={muon_lr})")
            print(f"   - Adam Group: {len(adam_params)} tensors (LR={adam_lr})")

        return Muon(param_groups)

    decay_params = [p for p in param_dict.values() if p.ndim >= 2]
    nodecay_params = [p for p in param_dict.values() if p.ndim < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    if opt_type == 'lion':
        return Lion(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    elif opt_type == 'sophia':
        return SophiaG(optim_groups, lr=lr, betas=(0.965, 0.99), rho=0.04, weight_decay=weight_decay)
    else:
        opt = optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
        opt.get_metrics = lambda: {}
        return opt

# END OF FILE: pretrain/components/optimizers.py