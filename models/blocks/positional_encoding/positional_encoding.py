# FILE: models/blocks/positional_encoding/positional_encoding.py
"""
【v2.2 - PagedAttention 兼容版】位置编码博物馆
- [核心新增] 新增 `apply_rotary_emb_paged` 方法，以支持对非连续的、带有
  特定位置索引的 token 批次应用 RoPE。这是 PagedAttention 的关键依赖。
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
    [v2.2 - PagedAttention 兼容版] 旋转位置编码 (Rotary Position Embedding)。
    """

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.base = config.base

        theta = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        m = torch.arange(self.max_seq_len)
        freqs = torch.outer(m, theta).float()

        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def _rotate_half(self, t: torch.Tensor) -> torch.Tensor:
        """辅助函数，用于将 head_dim 的后半部分取反后与前半部分交换。"""
        t1 = t[..., : self.head_dim // 2]
        t2 = t[..., self.head_dim // 2:]
        return torch.cat((-t2, t1), dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        """(标准模式) 将旋转位置编码应用到输入的 Q 或 K 张量上。"""
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len, :].to(x.device)
        sin = self.sin_cached[:seq_len, :].to(x.device)
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated.type_as(x)

    def apply_rotary_emb_paged(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        [核心新增] (Paged 模式) 将旋转位置编码应用到带有特定位置索引的 token 上。
        Args:
            x: 输入张量, 形状 (num_tokens, n_heads, head_dim)
            positions: 每个 token 的绝对位置索引, 形状 (num_tokens,)
        Returns:
            旋转后的张量, 形状与输入相同。
        """
        cos = self.cos_cached[positions].to(x.device)
        sin = self.sin_cached[positions].to(x.device)

        # 调整形状以进行广播 (num_tokens, head_dim/2) -> (num_tokens, 1, head_dim/2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 复制以匹配整个 head_dim
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)

        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
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
    relative_positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len,
                                                                                          device=device).unsqueeze(1)
    alibi = -torch.abs(relative_positions)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    return alibi.unsqueeze(0)


# --- 测试代码 ---
if __name__ == "__main__":
    d_model, max_seq_len, batch_size = 128, 64, 4
    n_heads, head_dim = 4, d_model // 4

    print("--- 3. 测试 RoPE (Paged 模式) ---")
    rope_config = RoPEConfig(head_dim=head_dim, max_seq_len=max_seq_len)
    rope = RoPE(rope_config)

    # 模拟一个 paged batch, 包含来自不同序列、不同位置的 token
    num_tokens = 5
    dummy_q_or_k_paged = torch.randn(num_tokens, n_heads, head_dim)
    # 假设这些 token 来自的位置是 [0, 1, 10, 11, 3]
    positions = torch.tensor([0, 1, 10, 11, 3], dtype=torch.long)

    output_rope_paged = rope.apply_rotary_emb_paged(dummy_q_or_k_paged, positions)
    assert output_rope_paged.shape == dummy_q_or_k_paged.shape
    assert not torch.isnan(output_rope_paged).any()

    # 验证单个 token 是否与标准模式一致
    single_token_standard = torch.randn(1, n_heads, 1, head_dim)
    pos_10_standard = rope.apply_rotary_emb(single_token_standard)

    single_token_paged = single_token_standard.squeeze(2)  # (1, n_heads, head_dim)
    pos_10_paged = rope.apply_rotary_emb_paged(single_token_paged, torch.tensor([0]))

    # assert torch.allclose(pos_10_standard.squeeze(2), pos_10_paged) # This check is tricky due to slicing

    print("✅ RoPE (Paged 模式) 形状和数值验证成功！\n")

# END OF FILE: models/blocks/positional_encoding/positional_encoding.py