# FILE: models/transformer.py
"""
将所有模块组装成一个完整的、现代化的Transformer语言模型。
"""
# --- 路径修复代码 ---
# 确保即使直接运行此脚本，也能找到项目根目录下的模块
import sys
import os
from pathlib import Path

# 将项目根目录添加到Python的模块搜索路径
# os.path.dirname(__file__) 获取当前文件所在目录 '.../models'
# Path(...).parent 获取上一级目录 '.../'
project_root = str(Path(os.path.dirname(__file__)).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 路径修复代码结束 ---

import torch
import torch.nn as nn
import math
from dataclasses import dataclass

from models.config import ModelArgs
from models.blocks.attention.attention import Attention, AttentionConfig
from models.blocks.feedforward.feedforward import FeedForward
from models.blocks.normalization.normalization import RMSNorm
from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig


class TransformerBlock(nn.Module):
    """
    一个Transformer块，包含注意力层和前馈网络层。
    采用 Pre-Normalization 架构。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 为Attention模块创建一个专用的配置对象
        attention_args = AttentionConfig(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
        )
        self.attention = Attention(attention_args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_hidden_dim,
            multiple_of=args.multiple_of
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, rope: RoPE) -> torch.Tensor:
        # 残差连接
        h = x + self.attention(self.attention_norm(x), mask, rope)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    完整的Transformer语言模型。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        self.args = args

        # 1. 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # 2. Transformer 块列表
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        # 3. 最终的归一化层和语言模型头
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 4. RoPE 位置编码模块
        # --- 核心修复：使用RoPEConfig对象而不是字典 ---
        rope_config = RoPEConfig(
            head_dim=args.dim // args.n_heads,
            max_seq_len=args.max_seq_len,
            base=args.rope_base
        )
        self.rope = RoPE(rope_config)

        # 5. 因果注意力mask
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bs, seq_len = tokens.shape
        assert seq_len <= self.args.max_seq_len, "输入序列长度超过模型最大长度限制"

        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(h, self.mask, self.rope)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


# --- 测试代码 ---
if __name__ == "__main__":
    from models.config import our_tutorial_model

    print("--- 测试完整的 Transformer 模型 ---")

    args = our_tutorial_model()

    hidden_dim = 4 * args.dim
    hidden_dim = int(2 * hidden_dim / 3)
    args.ffn_hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

    print("\n--- 模型配置 ---")
    print(args)

    model = Transformer(args).eval()

    batch_size = 4
    seq_len = args.max_seq_len // 2
    dummy_tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    print("\n--- 输入 ---")
    print(f"输入形状: {dummy_tokens.shape}")

    with torch.no_grad():
        logits = model(dummy_tokens)

    print("\n--- 输出 ---")
    print(f"输出Logits形状: {logits.shape}")

    expected_shape = (batch_size, seq_len, args.vocab_size)
    assert logits.shape == expected_shape, f"形状不匹配！期望 {expected_shape}, 得到 {logits.shape}"
    print("\n✅ 模型前向传播及形状验证成功！")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params / 1e6:.2f} M")

# END OF FILE: models/transformer.py
