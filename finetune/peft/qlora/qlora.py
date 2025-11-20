# FILE: finetune/peft/qlora/qlora.py
"""
[QLoRA Core] QLoRA 组装工厂。
功能：
1. 遍历模型，将所有 nn.Linear 替换为 Linear4bit。
2. 在 Linear4bit 之上应用 LoRA 适配器。
"""
import torch
import torch.nn as nn
import math
from typing import List
from .linear4bit import Linear4bit


class QLoRALayer(nn.Module):
    """
    一个组合层，包含：
    1. 冻结的 4-bit 基础层 (Linear4bit)
    2. 可训练的 LoRA 分支 (Adapter)
    """

    def __init__(
            self,
            base_layer: Linear4bit,
            rank: int,
            alpha: int,
            dropout: float
    ):
        super().__init__()
        self.base_layer = base_layer
        # 冻结基础层（虽然它本身也没有 Parameter，是 Buffer，但 bias 是 Parameter）
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA A: (in, r)
        # LoRA B: (r, out)
        # 为了匹配 F.linear(x, weight)，权重形状通常是 (out, in)
        # 所以 nn.Linear(in, rank) 的权重是 (rank, in)
        # 我们遵循 standard LoRA implementation: B @ A @ x

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 确保 LoRA 参数的数据类型与计算类型一致
        self.lora_A.to(base_layer.compute_dtype)
        self.lora_B.to(base_layer.compute_dtype)

    def forward(self, x: torch.Tensor):
        # 1. Base output (Quantized path)
        # Linear4bit 内部会处理解量化
        base_output = self.base_layer(x)

        # 2. LoRA output (Adapter path)
        # x -> Dropout -> A -> B -> Scale
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        return base_output + lora_output


def replace_linear_with_qlora(model: nn.Module, rank: int, alpha: int, dropout: float, target_modules: List[str],
                              compute_dtype=torch.bfloat16):
    """
    递归遍历模型，执行两步操作：
    1. 将目标 nn.Linear 转换为 Linear4bit (量化)。
    2. 用 QLoRALayer 包裹 Linear4bit (添加 LoRA)。
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # 递归
            replace_linear_with_qlora(module, rank, alpha, dropout, target_modules, compute_dtype)

        if isinstance(module, nn.Linear):
            # 检查是否在目标列表中
            # 例如 name 是 "wq"，target_modules 是 ["wq", "wk"]
            # 如果我们只看 leaf name，可能不够精确，但在简单 Transformer 中通常够用
            if any(t in name for t in target_modules):
                print(f"⚖️ Quantizing & Adapting: {name} -> QLoRA (4-bit + Adapter)")

                # 1. 转换为 4-bit
                linear4bit = Linear4bit.from_linear(module, block_size=64, compute_dtype=compute_dtype)

                # 2. 包裹 LoRA
                qlora_layer = QLoRALayer(linear4bit, rank, alpha, dropout)

                # 3. 替换
                setattr(model, name, qlora_layer)

                # 4. 释放原始 fp32 权重的显存
                del module
                torch.cuda.empty_cache() if torch.cuda.is_available() else None


def prepare_model_for_qlora_training(model: nn.Module):
    """
    准备训练：
    1. 冻结所有非 LoRA 参数。
    2. 确保只有 LoRA 参数 requires_grad=True。
    3. 打印可训练参数统计。
    """
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"\n📊 QLoRA Model Statistics:")
    print(f"   - Total Params: {all_param:,}")
    print(f"   - Trainable Params: {trainable_params:,}")
    print(f"   - Trainable Ratio: {trainable_params / all_param:.4%}")

# END OF FILE: finetune/peft/qlora/qlora.py