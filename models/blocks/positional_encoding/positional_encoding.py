# FILE: models/blocks/positional_encoding/positional_encoding.py
"""
【v2.1 - DDP兼容修复版】位置编码博物馆
- RoPE 实现已重构，移除了对 torch.complex64 的依赖，以兼容 DDP 的 gloo 后端。
"""
import torch
import torch.nn as nn
import math
from dataclasses import dataclass


# --- 1. 可学习的绝对位置编码 ---
class LearnedPositionalEncoding(nn.Module):
    """
    可学习的绝对位置编码。
    创建一个位置嵌入矩阵，在训练中被学习。
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个 (max_seq_len, d_model) 的嵌入矩阵
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # 创建位置ID: (seq_len) -> [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # -> (1, seq_len)
        # 查找位置嵌入并与x相加
        x = x + self.pe(positions)
        return self.dropout(x)


# --- 2. 三角函数绝对位置编码 ---
class SinusoidalPositionalEncoding(nn.Module):
    """
    三角函数绝对位置编码 (源自 "Attention Is All You Need")。
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将pe注册为buffer，这样它不会被视为模型参数，但会随模型移动(e.g., to(device))
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- 3. RoPE (旋转位置编码) ---
@dataclass
class RoPEConfig:
    head_dim: int = 64
    max_seq_len: int = 512
    base: int = 10000


class RoPE(nn.Module):
    """
    [v2.1 - DDP兼容与工业级对齐版] 旋转位置编码 (Rotary Position Embedding)。

    ### 设计哲学与DDP兼容性

    最初的RoPE实现为了代码的优雅性，使用了PyTorch的复数张量 `torch.complex64`。
    其核心思想是将 `head_dim` 维度的向量视为 `head_dim/2` 个复数，然后通过与
    `e^(i*m*theta)` 形式的位置编码复数相乘来高效地实现旋转。

    然而，在进行DDP（分布式数据并行）训练时，我们遇到了一个严重问题：用于CPU并行训练的
    `gloo` 后端不支持通过网络传输 `complex` 这种数据类型。这导致在DDP初始化同步模型状态时
    出现 `RuntimeError: Invalid scalar type`。

    为了解决此问题，并提升代码的普适性，我们重构了此实现。核心改动是将位置编码的复数
    `e^(iθ) = cos(θ) + i*sin(θ)` 拆分为两个独立的、使用标准 `float` 类型的 `cos_cached` 和
    `sin_cached` 缓冲区。`apply_rotary_emb` 方法也相应地从复数乘法修改为基于实数的等价
    数学运算 `(x * cos) + (rotate_half(x) * sin)`。

    ### 与工业级框架对齐

    这次重构不仅仅是为了兼容性，更是与主流工业级框架（如Hugging Face Transformers,
    Megatron-LM, vLLM）的最佳实践达成了完全一致。这些框架也普遍采用预计算 `cos/sin` 并
    进行实数运算的方式，原因如下：
    1.  **最大化兼容性**: `float` 类型是所有硬件（CPU, GPU, TPU）和分布式后端（gloo, nccl）
        都支持的“通用语言”，保证了代码的鲁棒性。
    2.  **底层优化友好**: 底层的CUDA/Triton内核开发者更倾向于直接操作`float`数组，这种显式
        的实数运算更容易进行内存访问和计算的极致优化。
    3.  **编译器友好**: 对`torch.compile`等JIT编译器而言，优化显式的实数运算比优化抽象的
        复数运算更加成熟和高效。

    因此，当前这个实现不仅解决了我们遇到的DDP问题，也使我们的代码达到了工业级的健壮性和标准。

    ### 性能分析 (理论)

    -   **复数实现**: `(a+bi) * (c+di) = (ac - bd) + (ad + bc)i`。每个复数元素乘法需要
        **4次实数乘法** 和 **2次实数加/减法**。
    -   **实数实现**: `(x * cos) + (rotate_half(x) * sin)`。每个`head_dim`中的元素对
        （对应一个复数）需要 **2次实数乘法** 和 **1次实数加法**，外加 `rotate_half` 操作
        （涉及切片、取反和拼接，主要是内存操作）。

    理论上，实数实现的浮点运算次数（FLOPs）更少。但在实际中，性能差异通常可以忽略不计，
    因为：
    a) `torch.complex` 的乘法可能由高度优化的库（如MKL, cuBLAS）在底层执行，效率很高。
    b) `rotate_half` 中的数据移动（尤其是`torch.cat`）会带来一些不可忽视的开销。

    最终，RoPE操作的总耗时远小于Attention中的大规模矩阵乘法（`Q @ K.T`），因此这点微小的
    性能差异对整体训练速度影响甚微。我们的首要考量是 **正确性、兼容性和代码清晰度**。
    """

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.base = config.base

        theta = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        m = torch.arange(self.max_seq_len)
        freqs = torch.outer(m, theta).float()

        # [核心修改] 不再使用 torch.polar 创建复数张量
        # 而是将 cos 和 sin 的值分别存储为浮点数缓冲区
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        将旋转位置编码应用到输入的 Q 或 K 张量上。
        Args:
            x: 输入张量, 形状 (bs, n_heads, seq_len, head_dim)
        Returns:
            旋转后的张量, 形状与输入相同。
        """

        # --- [核心修改] 使用基于实数的旋转实现 ---

        # 辅助函数，用于将 head_dim 的后半部分取反后与前半部分交换
        def rotate_half(t: torch.Tensor) -> torch.Tensor:
            t1 = t[..., : self.head_dim // 2]
            t2 = t[..., self.head_dim // 2:]
            return torch.cat((-t2, t1), dim=-1)

        # 获取当前序列长度所需的预计算值
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len, :].to(x.device)
        sin = self.sin_cached[:seq_len, :].to(x.device)

        # 调整形状以进行广播
        # (seq_len, head_dim/2) -> (1, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

        # 由于 cos 和 sin 只作用于成对的维度，我们需要将它们复制以匹配整个 head_dim
        # (1, 1, seq_len, head_dim/2) -> (1, 1, seq_len, head_dim)
        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)

        # RoPE 核心公式:
        # x_rotated = x * cos(m*theta) + rotate_half(x) * sin(m*theta)
        x_rotated = (x * cos) + (rotate_half(x) * sin)

        return x_rotated.type_as(x)


# --- 4. ALiBi (线性偏置) ---
def get_alibi_bias(n_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    生成ALiBi偏置矩阵。
    返回形状: (1, n_heads, seq_len, seq_len)
    """

    def get_slopes(n):
        def get_next_power_of_2(n):
            return 2 ** math.ceil(math.log2(n))

        m = torch.tensor(get_next_power_of_2(n))
        m = m.to(torch.float32)
        r = torch.arange(n)
        return (m ** (-2.0 ** (-math.log2(m) * (r + 1)))).tolist()

    slopes = torch.tensor(get_slopes(n_heads)).to(device)
    # 构造距离矩阵 (seq_len, seq_len)
    relative_positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
    # ALiBi 使用负的绝对距离
    alibi = -torch.abs(relative_positions)
    # (n_heads, seq_len, seq_len)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # (1, n_heads, seq_len, seq_len) for broadcasting
    return alibi.unsqueeze(0)


# --- 测试代码 ---
if __name__ == "__main__":
    d_model, max_seq_len, batch_size = 128, 64, 4
    dummy_input = torch.randn(batch_size, max_seq_len, d_model)

    print("--- 1. 测试 LearnedPositionalEncoding ---")
    learned_pe = LearnedPositionalEncoding(d_model, max_seq_len)
    output_lpe = learned_pe(dummy_input)
    assert output_lpe.shape == dummy_input.shape
    print("✅ Learned PE 形状验证成功！\n")

    print("--- 2. 测试 SinusoidalPositionalEncoding ---")
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_seq_len)
    output_spe = sinusoidal_pe(dummy_input)
    assert output_spe.shape == dummy_input.shape
    print("✅ Sinusoidal PE 形状验证成功！\n")

    print("--- 3. 测试 RoPE (DDP 兼容版) ---")
    n_heads, head_dim = 4, d_model // 4
    rope_config = RoPEConfig(head_dim=head_dim, max_seq_len=max_seq_len)
    rope = RoPE(rope_config)
    dummy_q_or_k = torch.randn(batch_size, n_heads, max_seq_len, head_dim)
    output_rope = rope.apply_rotary_emb(dummy_q_or_k)
    assert output_rope.shape == dummy_q_or_k.shape
    # 检查是否有NaN
    assert not torch.isnan(output_rope).any()
    print("✅ RoPE (DDP 兼容版) 形状和数值验证成功！\n")

    print("--- 4. 测试 ALiBi ---")
    alibi_bias = get_alibi_bias(n_heads, max_seq_len, device=torch.device('cpu'))
    expected_shape = (1, n_heads, max_seq_len, max_seq_len)
    assert alibi_bias.shape == expected_shape
    print(f"✅ ALiBi bias 形状验证成功: {alibi_bias.shape}")
    assert torch.all(torch.diagonal(alibi_bias[0, 0]) == 0)
    assert alibi_bias[0, 0, 0, 1] < 0 and alibi_bias[0, 0, 0, 1] > alibi_bias[0, 0, 0, 2]
    print("✅ ALiBi bias 数值特性验证成功！\n")

# END OF FILE: models/blocks/positional_encoding/positional_encoding.py