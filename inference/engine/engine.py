# FILE: inference/engine/engine.py
# -*- coding: utf-8 -*-
"""
[v2.0 - Prefix Caching Support]
- 实现多轮对话中的 KV Cache 复用。
- 当新 Prompt 包含历史前缀时，跳过 Prefill，直接复用 Cache。
"""
import torch
from typing import List, Dict
import logging

from models.transformer import Transformer
from tokenizers import Tokenizer
from inference.engine.kv_cache import KVCacheBase, StandardKVCache, LatentKVCache
from inference.strategies.sampling import sample
from inference.quantization import Quantizer


class InferenceEngine:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, quantize: bool = False):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # [Cache Store] 简单的 Session 缓存： {session_id: (tokens, kv_cache)}
        self.cache_store: Dict[str, tuple] = {}

        if quantize:
            if self.device.type != 'cpu':
                self.model.to('cpu')
                self.device = torch.device('cpu')
            self.model = Quantizer.quantize_dynamic(self.model)

    @torch.no_grad()
    def _prefill_with_cache(self, prompt_tokens: torch.Tensor, session_id: str = None):
        """
        智能 Prefill：检查是否有可复用的前缀。
        """
        batch_size, seq_len = prompt_tokens.shape
        assert batch_size == 1, "Prefix caching currently supports batch_size=1 only"

        cached_tokens, cached_kv = [], None
        start_pos = 0

        # 1. 尝试匹配 Cache
        if session_id and session_id in self.cache_store:
            stored_tokens, stored_kv = self.cache_store[session_id]
            stored_len = len(stored_tokens)

            # 找到最长公共前缀
            # current: [1, 2, 3, 4, 5]
            # stored : [1, 2, 3, 8, 9]
            # match  : [1, 2, 3] (len=3)

            min_len = min(seq_len, stored_len)
            match_len = 0
            # CPU list comparison is fast
            curr_list = prompt_tokens[0].tolist()
            for i in range(min_len):
                if curr_list[i] == stored_tokens[i]:
                    match_len += 1
                else:
                    break

            if match_len > 0:
                logging.info(f"🎯 Cache Hit! Reusing {match_len} tokens for session '{session_id}'.")
                cached_kv = stored_kv
                start_pos = match_len
            else:
                logging.info(f"💨 Cache Miss for session '{session_id}'.")

        # 2. 初始化或复用 KV Cache
        if cached_kv is None:
            kv_cache = self.model.create_kv_cache(max_batch_size=batch_size, device=self.device, dtype=self.dtype)
        else:
            kv_cache = cached_kv

        # 3. 计算新增部分的 Prefill
        if start_pos < seq_len:
            # 只计算新 token
            new_tokens = prompt_tokens[:, start_pos:]
            logits = self.model(new_tokens, kv_cache=kv_cache, start_pos=start_pos)
            last_logits = logits[:, -1, :]
        else:
            # 极端情况：Prompt 完全被 Cache 包含 (通常不会发生，除非重复输入)
            # 我们需要再跑一次最后一个 token 来获得 logits
            last_token = prompt_tokens[:, -1:]
            logits = self.model(last_token, kv_cache=kv_cache, start_pos=seq_len - 1)
            last_logits = logits[:, -1, :]

        return kv_cache, last_logits, seq_len

    def generate(
            self,
            prompts: List[str],
            max_new_tokens: int,
            temperature: float = 0.7,
            top_p: float = 0.9,
            session_id: str = None  # 新增 session_id
    ) -> List[str]:

        # Tokenize
        tokenized = self.tokenizer.encode(prompts[0])
        prompt_tokens = torch.tensor([tokenized.ids], device=self.device, dtype=torch.long)

        # Smart Prefill
        kv_cache, next_logits, current_pos = self._prefill_with_cache(prompt_tokens, session_id)

        # Decoding Loop
        generated_ids = []
        next_token = sample(next_logits, top_p=top_p, temperature=temperature).to(self.device)
        generated_ids.append(next_token.item())

        for _ in range(max_new_tokens - 1):
            # Decode step
            next_logits = self.model(next_token.view(1, 1), kv_cache=kv_cache, start_pos=current_pos)
            next_token = sample(next_logits[:, -1, :], top_p=top_p, temperature=temperature).to(self.device)
            generated_ids.append(next_token.item())
            current_pos += 1

            if next_token.item() == self.tokenizer.token_to_id("<|endoftext|>"):
                break

        # Decode Text
        full_ids = prompt_tokens[0].tolist() + generated_ids
        result_text = self.tokenizer.decode(full_ids)

        # 更新 Cache Store
        if session_id:
            self.cache_store[session_id] = (full_ids, kv_cache)

        return [result_text]

# END OF FILE: inference/engine/engine.py