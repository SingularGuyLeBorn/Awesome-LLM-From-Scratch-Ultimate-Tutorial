# FILE: models/transformer.py
"""
【v2.4 - DDP 验证死锁终极修复】
- 移除梯度检查点中的 `and self.training` 条件，使其在验证阶段也能生效，避免内存暴增导致的假死。
"""
import sys
import os
from pathlib import Path
from typing import Optional

project_root = str(Path(os.path.dirname(__file__)).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import math
import logging
from torch.utils.checkpoint import checkpoint

from models.config import ModelArgs
from models.blocks.attention.attention import Attention, AttentionConfig
from models.blocks.feedforward.feedforward import FeedForward
from models.blocks.normalization.normalization import RMSNorm
from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig
from inference.kv_cache import KVCache


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
        self.use_checkpointing = args.use_activation_checkpointing

    def _forward_impl(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: Optional[KVCache] = None,
                      start_pos: int = 0) -> torch.Tensor:
        """实际的前向传播逻辑"""
        h = x + self.attention(self.attention_norm(x), rope, layer_idx, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: Optional[KVCache] = None,
                start_pos: int = 0) -> torch.Tensor:
        """
        前向传播，根据配置决定是否使用梯度检查点。
        """
        # [核心修改] 移除了 `and self.training`
        # 在内存受限的环境中（比如我们的CPU项目），在验证阶段也启用梯度检查点是必要的。
        # 虽然这会带来不必要的重计算开销（因为验证不需要反向传播），但它能防止
        # 因为关闭检查点导致的激活值内存暴增，从而避免进程因使用虚拟内存而“假死”。
        # 我们的首要目标是“能跑完”，其次才是“跑得快”。
        if self.use_checkpointing:
            return checkpoint(
                lambda *inputs: self._forward_impl(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]),
                x, rope, layer_idx, kv_cache, start_pos,
                use_reentrant=False
            )
        else:
            return self._forward_impl(x, rope, layer_idx, kv_cache, start_pos)


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

        # 权重绑定
        self.tok_embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)

        if args.use_activation_checkpointing:
            logging.info("模型已配置为在训练和验证期间使用梯度检查点 (Activation Checkpointing)。")
        else:
            logging.info("梯度检查点未启用。")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module in [layer.attention.wo for layer in self.layers] or \
                    module in [layer.feed_forward.w_down for layer in self.layers]:
                module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * self.args.n_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if module.weight is self.lm_head.weight:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            tokens: torch.Tensor,
            return_hidden_states: bool = False,
            kv_cache: Optional[KVCache] = None,
            start_pos: int = 0
    ) -> torch.Tensor:
        bs, seq_len = tokens.shape

        if not kv_cache:
            assert seq_len <= self.args.max_seq_len, "输入序列长度超过模型最大长度限制"

        h = self.tok_embeddings(tokens)

        for i, layer in enumerate(self.layers):
            h = layer(h, self.rope, layer_idx=i, kv_cache=kv_cache, start_pos=start_pos)

        h = self.norm(h)

        if return_hidden_states:
            return h

        logits = self.lm_head(h)
        return logits
# END OF FILE: models/transformer.py