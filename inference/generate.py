# FILE: inference/generate.py
# -*- coding: utf-8 -*-
"""
[v1.1 - 生产级改造] 功能完备的 `.generate()` 方法
- 增加 eos_id 参数以支持提前终止。
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
        model,
        prompt_tokens,
        max_gen_len,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        eos_id=None  # [核心修改]
):
    model.eval()
    bs, prompt_len = prompt_tokens.shape
    device = prompt_tokens.device

    output_tokens = torch.zeros(bs, max_gen_len, dtype=torch.long, device=device)
    output_tokens[:, :prompt_len] = prompt_tokens

    # 跟踪每个序列是否已结束
    eos_reached = torch.zeros(bs, dtype=torch.bool, device=device)

    for cur_pos in range(prompt_len, max_gen_len):
        logits = model(output_tokens[:, :cur_pos])
        next_token = top_k_top_p_sampling(logits[:, -1, :], top_k=top_k, top_p=top_p, temperature=temperature)

        # 只更新尚未结束的序列
        output_tokens[~eos_reached, cur_pos] = next_token[~eos_reached].squeeze(-1)

        if eos_id is not None:
            eos_reached |= (next_token.squeeze(-1) == eos_id)
            if torch.all(eos_reached):
                break

    return output_tokens
# END OF FILE: inference/generate.py