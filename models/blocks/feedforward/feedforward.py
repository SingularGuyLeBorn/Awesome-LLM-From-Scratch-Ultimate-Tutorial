# FILE: models/blocks/feedforward/feedforward.py
"""
从零手写实现Transformer中的前馈网络 (FFN) 层。
这里我们直接实现现代LLM中流行的SwiGLU变体。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    一个采用SwiGLU激活函数的前馈网络模块。

    数学原理:
        FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    其中 SiLU(x) = x * sigmoid(x)
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        """
        Args:
            dim: 输入和输出的维度。
            hidden_dim: FFN的中间层维度。
            multiple_of: 一个整数，用于确保隐藏层维度是该数的倍数，以提高硬件效率。
                         这是LLaMA论文中提到的一个技巧。我们在配置中计算好后传入。
        """
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu 是 PyTorch 中 Swish/SiLU 激活函数的官方实现
        gate_output = F.silu(self.w_gate(x))
        up_output = self.w_up(x)

        # 逐元素相乘，实现门控机制
        fused_output = gate_output * up_output

        # 向下投影回原始维度
        output = self.w_down(fused_output)

        return output


# --- 测试代码 ---
if __name__ == "__main__":
    print("--- 测试SwiGLU FFN模块 ---")

    # --- 参数定义 ---
    batch_size = 4
    seq_len = 16
    dim = 128

    # 按照LLaMA的方式计算hidden_dim
    # 这是一个很好的实践，但为了模块的通用性，我们让调用者（模型配置）来做这件事
    multiple_of = 32
    # 经典 hidden_dim 是 4 * dim
    hidden_dim_classic = 4 * dim
    # SwiGLU 的 hidden_dim 大约是经典尺寸的 2/3
    hidden_dim_swiglu = int(2 * hidden_dim_classic / 3)
    # 向上取整到 multiple_of 的最接近的倍数
    hidden_dim = multiple_of * ((hidden_dim_swiglu + multiple_of - 1) // multiple_of)

    print(f"输入维度 (dim): {dim}")
    print(f"计算出的隐藏维度 (hidden_dim): {hidden_dim}")

    # --- 模块实例化和测试 ---
    ffn_module = FeedForward(dim=dim, hidden_dim=hidden_dim)

    # 创建随机输入
    random_input = torch.randn(batch_size, seq_len, dim)
    print("\n输入形状:", random_input.shape)

    # 前向传播
    output = ffn_module(random_input)
    print("输出形状:", output.shape)

    # --- 验证 ---
    # 最重要的验证是确保输出维度与输入维度一致
    assert output.shape == random_input.shape, "输出形状与输入形状不匹配！"
    print("\n✅ FFN模块形状验证成功！")

    # 打印参数量
    num_params = sum(p.numel() for p in ffn_module.parameters())
    print(f"FFN模块的参数量: {num_params / 1e6:.2f}M")

    # 对比经典FFN的参数量 (通常带偏置)
    classic_ffn_params = (dim * hidden_dim_classic + hidden_dim_classic) + (hidden_dim_classic * dim + dim)
    swiglu_ffn_params = (dim * hidden_dim) * 2 + (hidden_dim * dim)  # 3个无偏置矩阵
    print(f"近似经典FFN的参数量 (hidden_dim={hidden_dim_classic}): {classic_ffn_params / 1e6:.2f}M")
    print(f"SwiGLU FFN的参数量 (hidden_dim={hidden_dim}): {swiglu_ffn_params / 1e6:.2f}M")
    print("可以看到，通过调整hidden_dim，SwiGLU的参数量与经典FFN相当。")

# END OF FILE: models/blocks/feedforward/feedforward.py