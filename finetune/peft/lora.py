# FILE: finetune/peft/lora.py
"""
[v1.1 - 核心逻辑重构版] LoRA (Low-Rank Adaptation) 的核心实现。
- 彻底修复了 apply_lora_to_model 的递归和替换逻辑。
"""
import torch
import torch.nn as nn
import math
from typing import List


class LoRALayer(nn.Module):
    """一个包裹nn.Linear层的LoRA层。"""

    def __init__(
            self,
            base_layer: nn.Linear,
            rank: int,
            alpha: int,
            dropout: float,
    ):
        super().__init__()
        self.base_layer = base_layer
        in_features, out_features = base_layer.in_features, base_layer.out_features

        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        base_output = self.base_layer(x)
        lora_update = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_update


def apply_lora_to_model(model: nn.Module, rank: int, alpha: int, dropout: float, target_modules: List[str]):
    """
    [核心修复] 递归地遍历模型，将目标线性层替换为LoRALayer。
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(target_key in name for target_key in target_modules):
            # 这种情况通常发生在顶层，但我们的模型结构不会
            lora_layer = LoRALayer(module, rank, alpha, dropout)
            setattr(model, name, lora_layer)
            print(f"✅ 应用LoRA到顶层模块: {name} (rank={rank}, alpha={alpha})")

        # 递归进入子模块
        elif len(list(module.children())) > 0:
            # 检查子模块
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear) and any(target_key in sub_name for target_key in target_modules):
                    lora_layer = LoRALayer(sub_module, rank, alpha, dropout)
                    # 直接在父模块(module)上替换子模块(sub_module)
                    setattr(module, sub_name, lora_layer)
                    # 打印完整的路径
                    full_name = f"{name}.{sub_name}"
                    print(f"✅ 应用LoRA到: {full_name} (rank={rank}, alpha={alpha})")
                else:
                    # 继续深入递归
                    apply_lora_to_model(sub_module, rank, alpha, dropout, target_modules)


def freeze_base_model_for_lora(model: nn.Module):
    """
    冻结所有非LoRA参数，并计算可训练参数的比例。
    """
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    trainable_ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0
    print(f"\nLoRA 已应用。可训练参数: {trainable_params:,} (占总参数 {trainable_ratio:.2f}%)")
# END OF FILE: finetune/peft/lora.py