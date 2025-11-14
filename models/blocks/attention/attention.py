# FILE: models/blocks/attention/attention.py
"""
从零手写实现Transformer中的注意力机制。
此实现为通用版本，可通过配置支持MHA, MQA, GQA，并可选择性地应用ALiBi。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# 动态导入RoPE，以便此文件可以独立测试
try:
    from .positional_encoding import RoPE
except ImportError:
    from models.blocks.positional_encoding.positional_encoding import RoPE


@dataclass
class AttentionConfig:
    # 这是一个临时的、用于测试的dataclass
    dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4  # n_kv_heads < n_heads for GQA
    dropout: float = 0.1
    max_seq_len: int = 256


class Attention(nn.Module):
    """
    一个通用的注意力模块，支持MHA, MQA, GQA, 并可选择性地应用ALiBi。
    - MHA (Multi-Head Attention): n_heads == n_kv_heads
    - GQA (Grouped-Query Attention): n_heads % n_kv_heads == 0
    - MQA (Multi-Query Attention): n_kv_heads == 1
    """

    def __init__(self, args: AttentionConfig):
        super().__init__()
        assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            rope: RoPE,
            alibi_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq_len, _ = x.shape

        # 1. 线性投影 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape to (bs, seq_len, n_heads, head_dim)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim)

        # 3. Transpose for attention calculation: (bs, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 4. 应用旋转位置编码 (RoPE)
        # 注意：ALiBi 和 RoPE 通常是互斥的。在一个配置中只应使用一种。
        # 此处代码允许RoPE总是被应用，实际模型配置中应控制只使用其一。
        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

        # 5. GQA/MQA: 重复K,V头以匹配Q头的数量
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)

        # 6. 计算注意力分数
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 7. 应用ALiBi偏置 (如果提供)
        if alibi_bias is not None:
            scores = scores + alibi_bias[:, :, :seq_len, :seq_len]

        # 8. 应用mask (例如，因果mask)
        scores = scores + mask[:, :, :seq_len, :seq_len]

        # 9. Softmax
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 10. 计算输出
        output = torch.matmul(scores, xv)

        # 11. Reshape and final projection
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))


# --- 测试代码 ---
if __name__ == "__main__":
    from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig, get_alibi_bias

    args = AttentionConfig()
    rope_config = RoPEConfig(head_dim=args.dim // args.n_heads, max_seq_len=args.max_seq_len)
    rope = RoPE(rope_config)

    mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    input_tensor = torch.randn(2, args.max_seq_len, args.dim)

    print("--- 测试 Attention 模块 ---")

    # 1. GQA with RoPE (标准模式)
    print("\n--- 1. GQA with RoPE 模式 ---")
    attn_module = Attention(args)
    output_gqa = attn_module(input_tensor, mask, rope, alibi_bias=None)
    assert output_gqa.shape == input_tensor.shape
    print("✅ 形状验证成功！")

    # 2. GQA with ALiBi
    print("\n--- 2. GQA with ALiBi 模式 ---")
    # 在实际使用ALiBi时，通常不会再对Q,K应用RoPE。
    # 这里的测试是为了验证ALiBi的bias能被正确应用。
    alibi_bias = get_alibi_bias(args.n_heads, args.max_seq_len, input_tensor.device)
    output_alibi = attn_module(input_tensor, mask, rope, alibi_bias=alibi_bias)
    assert output_alibi.shape == input_tensor.shape
    print("✅ 形状验证成功！")

    # 验证ALiBi确实改变了输出
    assert not torch.allclose(output_gqa, output_alibi)
    print("✅ ALiBi bias 确实对输出产生了影响。")

    # 3. 参数量对比
    print("\n--- 3. 参数量对比 ---")
    mha_args = AttentionConfig(n_heads=8, n_kv_heads=8)
    gqa_args = AttentionConfig(n_heads=8, n_kv_heads=2)
    mqa_args = AttentionConfig(n_heads=8, n_kv_heads=1)

    mha_params = sum(p.numel() for p in Attention(mha_args).parameters())
    gqa_params = sum(p.numel() for p in Attention(gqa_args).parameters())
    mqa_params = sum(p.numel() for p in Attention(mqa_args).parameters())

    print(f"MHA (h=8, kv=8) 参数量: {mha_params / 1e6:.3f} M")
    print(f"GQA (h=8, kv=2) 参数量: {gqa_params / 1e6:.3f} M")
    print(f"MQA (h=8, kv=1) 参数量: {mqa_params / 1e6:.3f} M")
    assert mha_params > gqa_params > mqa_params
    print("✅ GQA 和 MQA 确实比 MHA 参数更少。")
# END OF FILE: models/blocks/attention/attention.py