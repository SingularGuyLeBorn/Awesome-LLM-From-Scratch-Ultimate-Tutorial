# FILE: models/transformer.py
"""
【v3.3 - RoPE 维度修复版】
- [核心修复] 修复 MLA 模式下 RoPE 初始化维度错误的问题。
  MLA 的 RoPE 只作用于 rope_head_dim (例如16)，而不是完整的 head_dim (例如32)。
"""
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

project_root = str(Path(os.path.dirname(__file__)).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import math
import logging
from torch.utils.checkpoint import checkpoint

from models.config import ModelArgs
from models.blocks.attention.attention import Attention
from models.blocks.feedforward.feedforward import FeedForward
from models.blocks.feedforward.moe import MoELayer
from models.blocks.normalization.normalization import RMSNorm
from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig
from inference.engine.kv_cache import KVCache


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attention = Attention(args)

        # MoE Logic
        is_moe_layer = (args.num_experts > 1) and (
                args.moe_layers_indices is None or layer_id in args.moe_layers_indices
        )

        if is_moe_layer:
            self.feed_forward = MoELayer(
                dim=args.dim,
                hidden_dim=args.ffn_hidden_dim,
                num_experts=args.num_experts,
                num_experts_per_tok=args.num_experts_per_tok,
                multiple_of=args.multiple_of
            )
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.ffn_hidden_dim,
                multiple_of=args.multiple_of
            )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_checkpointing = args.use_activation_checkpointing

    def _forward_impl(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: Optional[KVCache] = None,
                      start_pos: int = 0, paged_attention_inputs: Optional[Tuple] = None) -> torch.Tensor:
        attn_input = self.attention_norm(x)
        h = x + self.attention(
            attn_input, rope, layer_idx, kv_cache, start_pos, paged_attention_inputs=paged_attention_inputs
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: Optional[KVCache] = None,
                start_pos: int = 0, paged_attention_inputs: Optional[Tuple] = None) -> torch.Tensor:
        if self.use_checkpointing:
            return checkpoint(
                lambda *inputs: self._forward_impl(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]),
                x, rope, layer_idx, kv_cache, start_pos, paged_attention_inputs,
                use_reentrant=False
            )
        else:
            return self._forward_impl(x, rope, layer_idx, kv_cache, start_pos, paged_attention_inputs)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be set."
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args, layer_id=i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # [核心修复] 动态计算 RoPE 的维度
        # 如果是 MLA，RoPE 只作用于 decoupling 后的 PE 部分 (例如 16 维)
        # 否则 (MHA/GQA)，RoPE 作用于整个 head_dim (例如 128 维)
        if args.attention_variant == "mla":
            rope_dim = args.rope_head_dim
        else:
            rope_dim = args.dim // args.n_heads

        rope_config = RoPEConfig(
            head_dim=rope_dim,
            max_seq_len=args.max_seq_len,
            base=args.rope_base
        )
        self.rope = RoPE(rope_config)

        self.tok_embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)

        if args.use_activation_checkpointing:
            logging.info("Gradient Checkpointing: ENABLED")
        if args.num_experts > 1:
            logging.info(f"MoE Architecture: ENABLED ({args.num_experts} experts, top-{args.num_experts_per_tok})")
        logging.info(f"Attention Mechanism: {args.attention_variant.upper()} (RoPE dim: {rope_dim})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
            start_pos: int = 0,
            paged_attention_inputs: Optional[Tuple] = None,
    ) -> torch.Tensor:
        is_paged = paged_attention_inputs is not None
        if not is_paged:
            bs, seq_len = tokens.shape
            assert seq_len <= self.args.max_seq_len, f"Input sequence length ({seq_len}) exceeds limit ({self.args.max_seq_len})."

        h = self.tok_embeddings(tokens)

        for i, layer in enumerate(self.layers):
            h = layer(h, self.rope, layer_idx=i, kv_cache=kv_cache, start_pos=start_pos,
                      paged_attention_inputs=paged_attention_inputs)

        h = self.norm(h)

        if return_hidden_states:
            return h

        logits = self.lm_head(h)
        return logits

# END OF FILE: models/transformer.py