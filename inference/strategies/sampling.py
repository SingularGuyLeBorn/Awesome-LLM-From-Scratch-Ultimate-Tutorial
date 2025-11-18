# FILE: inference/strategies/sampling.py
# -*- coding: utf-8 -*-
"""
[v1.1 - 架构重构] 工业级推理采样器
- 已移动到 `inference/strategies/` 目录下。
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
) -> torch.Tensor:
    """
    一个健壮的采样函数，按正确顺序应用 temperature, top-k, top-p。

    Args:
        logits: 模型的原始输出 logits，形状为 (batch_size, vocab_size)。
        temperature: 温度系数。值越小，采样越倾向于高概率的token。
        top_k: Top-K 采样。只在概率最高的 K 个 token 中采样。
        top_p: Top-P (Nucleus) 采样。只在累积概率达到 P 的最小 token 集合中采样。

    Returns:
        下一个 token 的索引，形状为 (batch_size, 1)。
    """
    # 确保 logits 在 CPU 上进行处理，对于小批量这通常更快且避免同步
    logits = logits.to("cpu")

    # 1. 应用温度
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
    else:
        # 在 temperature=0 时，执行贪心采样
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. 应用 Top-K 采样
    if top_k > 0:
        # 取出 top_k 的概率值和索引
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        # 创建一个全零的概率分布，只在 top_k 的位置填充概率
        k_probs = torch.full_like(probs, 0)
        k_probs.scatter_(-1, top_k_indices, top_k_probs)
        probs = k_probs

    # 3. 应用 Top-P (Nucleus) 采样
    if top_p > 0:
        # 对概率进行排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找到累积概率超过 top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个超过阈值的位置也保留下来
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 创建一个 mask，将要移除的 token 位置标记为 True
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_indices,
                                                                               sorted_indices_to_remove)

        # 应用 mask，将不满足条件的 token 概率设为 0
        probs[indices_to_remove] = 0

    # 4. 重新归一化并采样
    # [核心修复] 必须重新归一化，否则 multinomial 会因概率和不为1而出错
    # 加上一个极小值避免除以零
    probs.div_(probs.sum(dim=-1, keepdim=True) + 1e-9)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token.to("cpu")  # 保持在CPU上
# END OF FILE: inference/strategies/sampling.py