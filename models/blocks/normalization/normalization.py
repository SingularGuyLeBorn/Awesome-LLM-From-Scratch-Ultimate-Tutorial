# FILE: models/blocks/normalization/normalization.py
"""
从零手写实现Transformer中的归一化层。
包含 BatchNorm, LayerNorm, RMSNorm, 以及 Qwen2RMSNorm (1+w技巧)。
"""

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """
    手写实现 BatchNorm。
    主要用于教学对比，通常不用于LLM。它在批次维度上进行归一化。
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # 可学习参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        # 用于推理的统计量
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状: (batch, seq_len, dim)
        if self.training:
            # 在 batch 和 seq_len 维度上计算均值和方差
            mean = x.mean(dim=[0, 1])
            var = x.var(dim=[0, 1], unbiased=False)

            # 更新 running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * x_norm + self.beta


class LayerNorm(nn.Module):
    """手写实现 Layer Normalization。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """手写实现 Root Mean Square Normalization。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x)


class Qwen2RMSNorm(nn.Module):
    """
    手写实现 Qwen2/Qwen3-next 中的 RMSNorm (1+w 技巧)。
    权重初始化为0，缩放因子为 (1 + weight)。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 关键：权重初始化为0
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键：使用 (1 + weight) 作为缩放因子
        return (1 + self.weight) * self._norm(x)


# --- 测试代码 ---
if __name__ == "__main__":
    print("--- 测试归一化层 ---")
    batch_size, seq_len, dim = 4, 16, 32
    random_input = torch.randn(batch_size, seq_len, dim)

    # 1. 测试 BatchNorm
    print("\n--- 测试 BatchNorm ---")
    my_bn = BatchNorm(dim)
    my_bn.train()  # 训练模式
    output_bn_train = my_bn(random_input)
    assert output_bn_train.shape == random_input.shape
    print("✅ BatchNorm (训练模式) 形状验证成功！")
    my_bn.eval()  # 推理模式
    output_bn_eval = my_bn(random_input)
    assert output_bn_eval.shape == random_input.shape
    print("✅ BatchNorm (推理模式) 形状验证成功！")

    # 2. 测试 LayerNorm
    print("\n--- 测试 LayerNorm ---")
    my_ln = LayerNorm(dim)
    output_my_ln = my_ln(random_input)
    pytorch_ln = nn.LayerNorm(dim)
    with torch.no_grad():
        pytorch_ln.weight.copy_(my_ln.gamma)
        pytorch_ln.bias.copy_(my_ln.beta)
    output_pytorch_ln = pytorch_ln(random_input)
    assert torch.allclose(output_my_ln, output_pytorch_ln, atol=1e-6)
    print("✅ LayerNorm 实现与 PyTorch 官方版本一致！")

    # 3. 测试 RMSNorm
    print("\n--- 测试 RMSNorm ---")
    my_rmsnorm = RMSNorm(dim)
    output_my_rmsnorm = my_rmsnorm(random_input)
    manual_rms_output = my_rmsnorm.weight * (
                random_input * torch.rsqrt(random_input.pow(2).mean(-1, keepdim=True) + my_rmsnorm.eps))
    assert torch.allclose(output_my_rmsnorm, manual_rms_output, atol=1e-6)
    print("✅ RMSNorm 实现正确！")

    # 4. 测试 Qwen2RMSNorm
    print("\n--- 测试 Qwen2RMSNorm (1+w trick) ---")
    my_qwen_rmsnorm = Qwen2RMSNorm(dim)
    output_my_qwen_rmsnorm = my_qwen_rmsnorm(random_input)
    # 验证权重是否初始化为0
    assert torch.sum(my_qwen_rmsnorm.weight.data) == 0
    print("   - 权重初始化为0, 验证成功。")
    # 验证计算公式
    manual_qwen_output = (1 + my_qwen_rmsnorm.weight) * (
                random_input * torch.rsqrt(random_input.pow(2).mean(-1, keepdim=True) + my_qwen_rmsnorm.eps))
    assert torch.allclose(output_my_qwen_rmsnorm, manual_qwen_output, atol=1e-6)
    print("✅ Qwen2RMSNorm 实现正确！")

# END OF FILE: models/blocks/normalization/normalization.py