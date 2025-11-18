# FILE: models/transformer.py
"""
【v3.4 - KVCache 适配版】
- Transformer.__init__ 现在根据 attention_variant 实例化正确的 KVCache (Standard vs Latent)。
- KVCache 实例现在作为模型的一部分，而不是在 generate 函数中临时创建。
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
# [核心修改] 导入所有 KVCache 类型
from inference.engine.kv_cache import KVCacheBase, StandardKVCache, LatentKVCache


class TransformerBlock(nn.Module):
    # ... (代码不变)
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attention = Attention(args)

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

    def _forward_impl(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: Optional[KVCacheBase] = None,
                      start_pos: int = 0, paged_attention_inputs: Optional[Tuple] = None) -> torch.Tensor:
        attn_input = self.attention_norm(x)
        h = x + self.attention(
            attn_input, rope, layer_idx, kv_cache, start_pos, paged_attention_inputs=paged_attention_inputs
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: Optional[KVCacheBase] = None,
                start_pos: int = 0, paged_attention_inputs: Optional[Tuple] = None) -> torch.Tensor:
        if self.training and self.use_checkpointing:
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

        # 日志
        if args.use_activation_checkpointing: logging.info("Gradient Checkpointing: ENABLED")
        if args.num_experts > 1: logging.info(
            f"MoE Architecture: ENABLED ({args.num_experts} experts, top-{args.num_experts_per_tok})")
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
            kv_cache: Optional[KVCacheBase] = None,
            start_pos: int = 0,
            paged_attention_inputs: Optional[Tuple] = None,
    ) -> torch.Tensor:
        is_paged = paged_attention_inputs is not None
        if not is_paged:
            _bsz, seqlen = tokens.shape
            assert seqlen <= self.args.max_seq_len, f"Cannot forward sequence of length {seqlen}, max is {self.args.max_seq_len}"

        h = self.tok_embeddings(tokens)

        for i, layer in enumerate(self.layers):
            h = layer(h, self.rope, layer_idx=i, kv_cache=kv_cache, start_pos=start_pos,
                      paged_attention_inputs=paged_attention_inputs)

        h = self.norm(h)

        if return_hidden_states:
            return h

        logits = self.lm_head(h)
        return logits

    # [新增] 辅助函数，用于在推理前创建正确的 KVCache
    def create_kv_cache(self, max_batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCacheBase:
        if self.args.attention_variant == "mla":
            return LatentKVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.args.max_seq_len,
                n_layers=self.args.n_layers,
                kv_lora_rank=self.args.kv_lora_rank,
                rope_head_dim=self.args.rope_head_dim,
                device=device,
                dtype=dtype
            )
        else:  # mha, gqa, mqa, linear, moba
            return StandardKVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.args.max_seq_len,
                n_layers=self.args.n_layers,
                n_kv_heads=self.args.n_kv_heads,
                head_dim=self.args.dim // self.args.n_heads,
                device=device,
                dtype=dtype
            )

# END OF FILE: models/transformer.py