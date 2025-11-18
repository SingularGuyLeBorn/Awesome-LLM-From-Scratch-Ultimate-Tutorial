# FILE: inference/generate.py
# -*- coding: utf-8 -*-
"""
[v2.7 - KVCache 适配]
- generate 函数不再直接创建 KVCache。
- 改为调用 model.create_kv_cache()，以支持 Standard 和 Latent 两种缓存类型。
"""
import torch
from typing import Iterator

from models.transformer import Transformer
from inference.strategies.sampling import sample


@torch.no_grad()
def generate(
        model: Transformer,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = None
) -> torch.Tensor:
    model.eval()
    bs, prompt_len = prompt_tokens.shape
    device = prompt_tokens.device
    dtype = next(model.parameters()).dtype

    # [核心修改] 调用模型工厂方法创建合适的 KV Cache
    kv_cache = model.create_kv_cache(max_batch_size=bs, device=device, dtype=dtype)

    # Prefill
    model(prompt_tokens, kv_cache=kv_cache, start_pos=0)

    # Decode
    eos_reached = torch.zeros(bs, dtype=torch.bool, device=device)
    all_tokens = [prompt_tokens]
    current_token = prompt_tokens[:, -1].view(bs, 1)

    total_len = min(model.args.max_seq_len, prompt_len + max_new_tokens)

    for cur_pos in range(prompt_len, total_len):
        logits = model(current_token, kv_cache=kv_cache, start_pos=cur_pos - 1)
        next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p).to(device)

        all_tokens.append(next_token)
        current_token = next_token

        if eos_id is not None:
            eos_reached |= (next_token.squeeze(-1) == eos_id)
            if torch.all(eos_reached):
                break

    return torch.cat(all_tokens, dim=1)


@torch.no_grad()
def generate_stream(
        model: Transformer,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = None
) -> Iterator[int]:
    model.eval()
    bs, prompt_len = prompt_tokens.shape
    assert bs == 1, "Streaming only supports batch size of 1."
    device = prompt_tokens.device
    dtype = next(model.parameters()).dtype

    # [核心修改]
    kv_cache = model.create_kv_cache(max_batch_size=bs, device=device, dtype=dtype)

    # Prefill
    model(prompt_tokens, kv_cache=kv_cache, start_pos=0)

    # Decode
    current_token = prompt_tokens[:, -1].view(bs, 1)
    total_len = min(model.args.max_seq_len, prompt_len + max_new_tokens)

    for cur_pos in range(prompt_len, total_len):
        logits = model(current_token, kv_cache=kv_cache, start_pos=cur_pos - 1)
        next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p).to(device)

        current_token = next_token

        if eos_id is not None and next_token.item() == eos_id:
            break

        yield next_token.item()
# END OF FILE: inference/generate.py