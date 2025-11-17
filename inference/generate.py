# FILE: inference/generate.py
# -*- coding: utf-8 -*-
"""
[v2.3 - 语义净化版]
- 核心修复：将 `max_gen_len` 参数重命名为 `max_new_tokens`，以消除语义歧义并修复 PPO rollout 中的 silent bug。
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Iterator

from models.transformer import Transformer, ModelArgs
from inference.kv_cache import KVCache


def top_k_top_p_sampling(logits, top_k=50, top_p=1.0, temperature=1.0):
    logits = logits / temperature
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, top_k)
        kth_vals = top_k_vals[:, -1].unsqueeze(-1)
        logits[logits < kth_vals] = -float('Inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate(
        model: Transformer,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,  # [核心修改] 参数名变更，语义更清晰
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

    # [核心修改] 确保生成的总长度不超过模型的最大长度限制
    total_len = min(max_seq_len, prompt_len + max_new_tokens)

    for cur_pos in range(prompt_len, total_len):
        logits = model(current_token, kv_cache=kv_cache, start_pos=cur_pos - 1)
        next_token = top_k_top_p_sampling(logits[:, -1, :], top_k, top_p, temperature)

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
        max_new_tokens: int,  # [核心修改] 参数名变更
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

    # [核心修改] 确保生成的总长度不超过模型的最大长度限制
    total_len = min(max_seq_len, prompt_len + max_new_tokens)

    for cur_pos in range(prompt_len, total_len):
        logits = model(current_token, kv_cache=kv_cache, start_pos=cur_pos - 1)
        next_token = top_k_top_p_sampling(logits[:, -1, :], top_k, top_p, temperature)

        current_token = next_token

        if eos_id is not None and next_token.item() == eos_id:
            break

        yield next_token.item()
# END OF FILE: inference/generate.py