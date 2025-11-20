# FILE: finetune/peft/qlora/nf4.py
# -*- coding: utf-8 -*-
"""
[QLoRA Core] NF4 (NormalFloat 4-bit) 数据类型的纯 PyTorch 实现。

理论背景:
NF4 是一种信息论上最优的 4-bit 数据类型，专门用于保存正态分布的数据（如神经网络权重）。
它通过分位数函数 (Quantile Function) 将标准正态分布 N(0, 1) 映射到 [-1, 1] 范围内的 16 个离散值。

实现细节:
1. get_nf4_codebook: 生成 NF4 的 16 个标准值。
2. quantize_blockwise: 执行分块量化 (Block-wise Quantization)，通常 block_size=64。
3. dequantize_blockwise: 执行反量化，恢复为 BF16/FP32。
4. pack_4bit / unpack_4bit: 利用位运算将两个 4-bit 索引打包进一个 uint8 字节。
"""
import torch


# [修正] 移除了未使用的 scipy 和 numpy 导入，保持纯 PyTorch 实现
# from scipy.stats import norm
# import numpy as np

def get_nf4_codebook(device='cpu'):
    """
    生成 NF4 的码本 (Codebook)。
    基于标准正态分布的分位数。
    """
    # 实际上 QLoRA 论文给出的标准 NF4 值如下：
    # 这些值是通过 scipy.stats.norm.ppf(quantiles) 计算得出的，我们直接硬编码以减少依赖。
    nf4_data = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ]
    # 转换为 tensor
    return torch.tensor(nf4_data, dtype=torch.float32, device=device)


def pack_4bit(q_indices: torch.Tensor) -> torch.Tensor:
    """
    将 4-bit 索引打包到 uint8 张量中。
    输入: (N,), uint8, 范围 [0, 15]
    输出: (N // 2,), uint8
    """
    if q_indices.numel() % 2 != 0:
        raise ValueError("打包时元素总数必须是偶数")

    # 确保是 uint8
    q_indices = q_indices.to(torch.uint8)

    # 分离高位和低位
    # 假设 input = [a, b, c, d, ...]
    # 我们将其 reshape 为 (-1, 2) -> [[a, b], [c, d], ...]
    reshaped = q_indices.reshape(-1, 2)

    # high (4 bits) << 4 | low (4 bits)
    # 第一个元素放高位，第二个放低位
    high = reshaped[:, 0]
    low = reshaped[:, 1]

    packed = (high << 4) | low
    return packed


def unpack_4bit(packed_tensor: torch.Tensor, original_numel: int) -> torch.Tensor:
    """
    将 uint8 张量解包为 4-bit 索引。
    """
    # high
    high = (packed_tensor >> 4) & 0xF
    # low
    low = packed_tensor & 0xF

    # stack: (N/2, 2) -> flatten -> (N,)
    unpacked = torch.stack([high, low], dim=-1).reshape(-1)

    # 可能不需要这一步，如果 original_numel 总是偶数
    return unpacked[:original_numel]


def quantize_blockwise(x: torch.Tensor, block_size: int = 64):
    """
    对输入张量 x 进行 Block-wise NF4 量化。

    Args:
        x: 输入权重 (FP32/BF16/FP16)
        block_size: 量化块大小 (通常 64)

    Returns:
        packed_indices: 打包后的 uint8 索引
        absmax: 每个块的缩放因子 (FP32)
    """
    # 1. 展平并 Padding
    original_shape = x.shape
    numel = x.numel()

    # 确保能被 block_size 整除
    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        x_flat = torch.nn.functional.pad(x.flatten(), (0, pad_len))
    else:
        x_flat = x.flatten()

    # 2. Reshape 为 (num_blocks, block_size)
    blocks = x_flat.view(-1, block_size)

    # 3. 计算 AbsMax (每个 block 的最大绝对值)
    absmax = blocks.abs().max(dim=1, keepdim=True)[0]
    # 防止除以 0
    absmax = absmax.clamp(min=1e-5)

    # 4. 归一化到 [-1, 1]
    blocks_norm = blocks / absmax

    # 5. 映射到 NF4 码本
    codebook = get_nf4_codebook(x.device)

    # 暴力搜索最近邻 (Nearest Neighbor Search)
    # blocks_norm: (N_blocks, 64) -> (N_blocks, 64, 1)
    # codebook: (16,) -> (1, 1, 16)
    # distance: (N_blocks, 64, 16)
    # 这种方式虽然占用显存，但不需要写 CUDA kernel
    # 为了省显存，我们可以分批做

    indices_list = []
    # 小批次处理以防 OOM
    sub_batch_size = 1024
    for i in range(0, blocks_norm.shape[0], sub_batch_size):
        sub_blocks = blocks_norm[i: i + sub_batch_size].unsqueeze(-1)  # (B, 64, 1)
        dist = (sub_blocks - codebook.view(1, 1, -1)).abs()  # (B, 64, 16)
        sub_indices = dist.argmin(dim=-1)  # (B, 64)
        indices_list.append(sub_indices)

    indices = torch.cat(indices_list, dim=0)  # (num_blocks, 64)

    # 6. 打包
    indices_flat = indices.flatten()
    packed_indices = pack_4bit(indices_flat)

    # 如果有 padding，我们需要记录原始长度以便解包时截断，但在 Linear 层中通常维度是固定的
    # 这里我们假设外部会处理 shape，只返回核心数据

    return packed_indices, absmax.flatten(), original_shape, pad_len


def dequantize_blockwise(packed_indices: torch.Tensor, absmax: torch.Tensor, original_shape: tuple, pad_len: int,
                         block_size: int = 64, target_dtype=torch.float32):
    """
    反量化过程。
    """
    device = packed_indices.device

    # 1. 解包索引
    # indices: (total_padded_elements,)
    # absmax: (num_blocks,)

    # 算出包含 padding 的总元素数
    total_elements = packed_indices.numel() * 2
    indices = unpack_4bit(packed_indices, total_elements)

    # 2. 查表 (Codebook Lookup)
    codebook = get_nf4_codebook(device).to(target_dtype)
    # (total_padded_elements,)
    values = codebook[indices.long()]

    # 3. 恢复尺度 (Rescale)
    # values: (num_blocks * 64,)
    # absmax: (num_blocks,) -> repeat -> (num_blocks * 64,)

    # reshape values to (num_blocks, block_size)
    values_blocked = values.view(-1, block_size)
    absmax_expanded = absmax.unsqueeze(1).to(target_dtype)

    dequantized = values_blocked * absmax_expanded

    # 4. 移除 Padding 并恢复形状
    dequantized_flat = dequantized.flatten()
    if pad_len > 0:
        dequantized_flat = dequantized_flat[:-pad_len]

    return dequantized_flat.view(original_shape)


# --- 简单的单元测试 ---
if __name__ == "__main__":
    print("Testing NF4 Quantization...")
    torch.manual_seed(42)
    # 模拟一个权重矩阵 (128, 128)
    original_weight = torch.randn(128, 128)

    print(f"Original Weight (FP32): {original_weight.element_size() * original_weight.numel() / 1024:.2f} KB")

    # 量化
    packed, absmax, shape, pad = quantize_blockwise(original_weight, block_size=64)

    # 计算压缩后大小
    packed_size = packed.element_size() * packed.numel()
    absmax_size = absmax.element_size() * absmax.numel()
    total_size = packed_size + absmax_size
    print(f"Quantized (Packed + Absmax): {total_size / 1024:.2f} KB")
    print(f"Compression Ratio: {(original_weight.element_size() * original_weight.numel()) / total_size :.2f}x")

    # 反量化
    recovered_weight = dequantize_blockwise(packed, absmax, shape, pad, block_size=64)

    # 验证误差
    error = (original_weight - recovered_weight).abs().mean()
    print(f"Mean Reconstruction Error: {error.item():.6f}")

    assert recovered_weight.shape == original_weight.shape
    print("✅ Test Passed!")

# END OF FILE: finetune/peft/qlora/nf4.py