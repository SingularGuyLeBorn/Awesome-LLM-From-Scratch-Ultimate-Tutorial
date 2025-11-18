# FILE: inference/generate.py
# -*- coding: utf-8 -*-
"""
[v2.6 - 架构重构] 基础的、一体化的 token 生成器。

此文件中的 `generate` 和 `generate_stream` 函数提供了一个简单、自包含的
推理实现。它们在一个函数内部混合了 Prefill 和 Decode 逻辑。

对于更结构化、更清晰地展示“PD分离”思想的推理架构，请参考 `inference/engine/engine.py`。
那个文件中的 `InferenceEngine` 是一个更接近工业级推理引擎设计的迷你实现。
"""
import torch
from typing import Iterator

# [核心修改] 更新导入路径
from models.transformer import Transformer
from inference.engine.kv_cache import KVCache
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
    max_seq_len = model.args.max_seq_len
    device = prompt_tokens.device
    dtype = model.tok_embeddings.weight.dtype

    kv_cache = KVCache(
        max_batch_size=bs, max_seq_len=max_seq_len, n_layers=model.args.n_layers,
        n_kv_heads=model.args.n_kv_heads, head_dim=model.args.dim // model.args.n_heads,
        device=device, dtype=dtype
    )

    model(prompt_tokens, kv_cache=kv_cache, start_pos=0)

    eos_reached = torch.zeros(bs, dtype=torch.bool, device=device)
    all_tokens = [prompt_tokens]
    current_token = prompt_tokens[:, -1].view(bs, 1)

    total_len = min(max_seq_len, prompt_len + max_new_tokens)

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

    assert bs == 1, "Streaming generation only supports a batch size of 1."

    max_seq_len = model.args.max_seq_len
    device = prompt_tokens.device
    dtype = model.tok_embeddings.weight.dtype

    kv_cache = KVCache(
        max_batch_size=bs, max_seq_len=max_seq_len, n_layers=model.args.n_layers,
        n_kv_heads=model.args.n_kv_heads, head_dim=model.args.dim // model.args.n_heads,
        device=device, dtype=dtype
    )

    model(prompt_tokens, kv_cache=kv_cache, start_pos=0)

    current_token = prompt_tokens[:, -1].view(bs, 1)

    total_len = min(max_seq_len, prompt_len + max_new_tokens)

    for cur_pos in range(prompt_len, total_len):
        logits = model(current_token, kv_cache=kv_cache, start_pos=cur_pos - 1)
        next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p).to(device)

        current_token = next_token

        if eos_id is not None and next_token.item() == eos_id:
            break

        yield next_token.item()
# END OF FILE: inference/generate.py