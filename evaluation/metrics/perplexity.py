# FILE: evaluation/metrics/perplexity.py
"""
实现语言模型评估的核心指标：Perplexity (困惑度)。
"""
import torch


def calculate_perplexity(loss: float) -> float:
    """
    根据给定的损失值计算困惑度。

    Perplexity 是交叉熵损失的指数形式。
    PPL = exp(loss)

    一个更低的困惑度意味着模型对于预测下一个词更有信心，因此模型性能更好。

    Args:
        loss (float): 模型在评估集上的平均交叉熵损失。

    Returns:
        float: 对应的困惑度值。
    """
    try:
        # 确保损失不是负数
        if loss < 0:
            return float('inf')
        return torch.exp(torch.tensor(loss)).item()
    except OverflowError:
        # 如果loss非常大，exp(loss)可能会溢出
        return float('inf')


# --- 测试代码 ---
if __name__ == "__main__":
    print("--- 测试 Perplexity 计算 ---")

    # 场景1: 低损失 -> 低困惑度 (好模型)
    low_loss = 2.5
    ppl_low = calculate_perplexity(low_loss)
    print(f"Loss: {low_loss:.4f} -> Perplexity: {ppl_low:.4f}")
    assert 12.1 < ppl_low < 12.2

    # 场景2: 高损失 -> 高困惑度 (差模型)
    high_loss = 5.0
    ppl_high = calculate_perplexity(high_loss)
    print(f"Loss: {high_loss:.4f} -> Perplexity: {ppl_high:.4f}")
    assert 148.4 < ppl_high < 148.5

    # 场景3: 零损失 -> 困惑度为1 (完美模型)
    zero_loss = 0.0
    ppl_zero = calculate_perplexity(zero_loss)
    print(f"Loss: {zero_loss:.4f} -> Perplexity: {ppl_zero:.4f}")
    assert ppl_zero == 1.0

    print("\n✅ Perplexity 计算验证成功！")

# END OF FILE: evaluation/metrics/perplexity.py