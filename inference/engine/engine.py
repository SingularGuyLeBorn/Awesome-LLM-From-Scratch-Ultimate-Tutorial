# FILE: inference/engine/engine.py
# -*- coding: utf-8 -*-
"""
[v1.2 - 架构重构] 一个简单的、五脏俱全的批处理推理引擎。

本引擎的核心设计思想是“PD分离”，即 Prompt Processing (Prefill) 和
Token Generation (Decode) 两个阶段的分离。
"""
import torch
from typing import List

# [核心修改] 更新导入路径
from models.transformer import Transformer
from tokenizers import Tokenizer
from inference.engine.kv_cache import KVCache
from inference.strategies.sampling import sample


class InferenceEngine:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def _prefill(self, prompts_tokens: torch.Tensor, attention_mask: torch.Tensor) -> (KVCache, torch.Tensor):
        """
        P - 预填充/提示处理 (Prefill / Prompt Processing) 阶段。
        一次性处理整个输入prompt，生成初始的KV Cache。
        """
        batch_size, seq_len = prompts_tokens.shape

        kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=self.model.args.max_seq_len,
            n_layers=self.model.args.n_layers,
            n_kv_heads=self.model.args.n_kv_heads,
            head_dim=self.model.args.dim // self.model.args.n_heads,
            device=self.device,
            dtype=self.dtype
        )

        logits = self.model(prompts_tokens, kv_cache=kv_cache, start_pos=0)

        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        last_logits = logits[torch.arange(batch_size), last_token_indices, :]

        return kv_cache, last_logits

    @torch.no_grad()
    def _decode(self, last_tokens: torch.Tensor, kv_cache: KVCache, current_pos: int) -> torch.Tensor:
        """
        D - 解码/生成 (Decode / Generation) 阶段。
        基于现有的KV Cache，处理单个新token，生成下一个token的logits。
        """
        logits = self.model(last_tokens, kv_cache=kv_cache, start_pos=current_pos - 1)
        return logits[:, -1, :]

    def generate(
            self,
            prompts: List[str],
            max_new_tokens: int,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ) -> List[str]:
        """
        批量处理推理请求。
        """
        self.tokenizer.enable_padding(direction='left', pad_id=self.tokenizer.token_to_id("<|pad|>"),
                                      pad_token="<|pad|>")
        tokenized_prompts = self.tokenizer.encode_batch(prompts)
        self.tokenizer.no_padding()

        prompts_tokens = torch.tensor([p.ids for p in tokenized_prompts], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor([p.attention_mask for p in tokenized_prompts], device=self.device,
                                      dtype=torch.long)

        batch_size, prompt_len = prompts_tokens.shape

        kv_cache, next_logits = self._prefill(prompts_tokens, attention_mask)

        generated_tokens = []
        next_token = sample(next_logits, top_p=top_p, temperature=temperature).to(self.device)
        generated_tokens.append(next_token)

        for i in range(1, max_new_tokens):
            current_pos = prompt_len + i
            if current_pos >= self.model.args.max_seq_len:
                break

            next_logits = self._decode(next_token, kv_cache, current_pos)
            next_token = sample(next_logits, top_p=top_p, temperature=temperature).to(self.device)
            generated_tokens.append(next_token)

        all_generated_tokens = torch.cat(generated_tokens, dim=1)
        results = self.tokenizer.decode_batch(all_generated_tokens.tolist())

        return [prompt + result for prompt, result in zip(prompts, results)]

# END OF FILE: inference/engine/engine.py