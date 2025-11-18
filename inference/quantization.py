# FILE: inference/quantization.py
# -*- coding: utf-8 -*-
"""
[v1.0 - 动态量化实现] CPU推理加速模块
功能: 提供将 PyTorch 模型动态量化为 Int8 的工具。
这对于在内存受限（如 16GB RAM）的 CPU 设备上运行较大模型（>1B）至关重要。
"""
import torch
import torch.nn as nn
import time
import logging


class Quantizer:
    """
    动态量化器。
    目前支持 PyTorch 原生的动态量化 (Dynamic Quantization)。
    它将 nn.Linear 层的权重转换为 int8，但在计算时会将激活值保持为浮点数（或动态量化），
    从而显著减少模型大小并加速 CPU 推理。
    """

    @staticmethod
    def quantize_dynamic(model: nn.Module, dtype=torch.qint8) -> nn.Module:
        """
        对模型应用动态量化。

        Args:
            model: 待量化的 PyTorch 模型 (通常是 float32 或 bfloat16)。
            dtype: 目标量化类型，通常是 torch.qint8。

        Returns:
            nn.Module: 量化后的模型。
        """
        start_time = time.perf_counter()
        logging.info(f"⚖️ 正在对模型应用动态量化 (Target: {dtype})...")

        # PyTorch 动态量化主要针对 Linear 和 LSTM/GRU/RNN 层
        # 我们主要关注 Linear 层
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # 只量化线性层
            dtype=dtype
        )

        end_time = time.perf_counter()
        logging.info(f"✅ 动态量化完成。耗时: {end_time - start_time:.2f}s")
        return quantized_model

    @staticmethod
    def print_model_size(model: nn.Module, name: str = "Model"):
        """
        打印模型的参数大小（以 MB 为单位）。
        """
        torch.save(model.state_dict(), "temp.p")
        size_mb = os.path.getsize("temp.p") / 1e6
        print(f"📦 {name} Size: {size_mb:.2f} MB")
        os.remove("temp.p")


if __name__ == "__main__":
    # 简单的测试用例
    import os

    print("--- 测试动态量化 ---")
    # 创建一个简单的线性模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(1024, 256)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleModel()
    print("原始模型结构:")
    print(model)
    Quantizer.print_model_size(model, "Original Model")

    # 应用量化
    q_model = Quantizer.quantize_dynamic(model)
    print("\n量化后模型结构:")
    print(q_model)
    Quantizer.print_model_size(q_model, "Quantized Model")

    # 验证推理
    input_tensor = torch.randn(1, 512)
    with torch.no_grad():
        out_orig = model(input_tensor)
        out_quant = q_model(input_tensor)

    # 注意：量化会有精度损失，所以 assert allclose 可能会失败，这里只打印差值
    diff = (out_orig - out_quant).abs().mean().item()
    print(f"\n平均输出差异: {diff:.6f}")
    print("✅ 测试完成。")

# END OF FILE: inference/quantization.py