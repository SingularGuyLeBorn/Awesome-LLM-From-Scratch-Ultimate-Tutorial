# FILE: models/transformer.py
"""
【v2.0 - 架构增强版】
- `forward` 方法新增 `return_hidden_states` 参数，使其更加灵活。
"""
import sys
import os
from pathlib import Path

project_root = str(Path(os.path.dirname(__file__)).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import math
import logging

from models.config import ModelArgs
from models.blocks.attention.attention import Attention, AttentionConfig
from models.blocks.feedforward.feedforward import FeedForward
from models.blocks.normalization.normalization import RMSNorm
from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        attention_args = AttentionConfig(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            is_causal=True
        )
        self.attention = Attention(attention_args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_hidden_dim,
            multiple_of=args.multiple_of
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), rope)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be set."
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        rope_config = RoPEConfig(
            head_dim=args.dim // args.n_heads,
            max_seq_len=args.max_seq_len,
            base=args.rope_base
        )
        self.rope = RoPE(rope_config)

        self.apply(self._init_weights)
        logging.info("模型权重已根据SOTA实践进行初始化。")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module in [layer.attention.wo for layer in self.layers] or \
                    module in [layer.feed_forward.w_down for layer in self.layers]:
                module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * self.args.n_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, return_hidden_states: bool = False) -> torch.Tensor:
        bs, seq_len = tokens.shape
        assert seq_len <= self.args.max_seq_len, "输入序列长度超过模型最大长度限制"

        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, self.rope)
        h = self.norm(h)

        # [核心修复] 根据参数决定是否返回隐藏状态
        if return_hidden_states:
            return h

        logits = self.lm_head(h)
        return logits
# END OF FILE: models/transformer.py