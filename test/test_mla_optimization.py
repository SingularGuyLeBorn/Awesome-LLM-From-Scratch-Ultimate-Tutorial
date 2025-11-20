# FILE: tests/test_mla_optimization.py
"""
验证 MLA (Multi-Head Latent Attention) 的矩阵吸收优化是否正确。
对比 Naive 实现（训练路径）和 Optimized 实现（推理路径）的输出数值。
"""
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from models.blocks.attention.standard import MultiHeadLatentAttention
from models.config import ModelArgs
from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig
from inference.engine.kv_cache import LatentKVCache


def test_mla_correctness():
    print("🧪 Testing MLA Matrix Absorption Optimization...")

    # 1. 配置
    args = ModelArgs(
        dim=128,
        n_heads=4,
        n_kv_heads=4,  # MLA typically ignored n_kv_heads as it's decoupled
        q_lora_rank=64,
        kv_lora_rank=32,
        nope_head_dim=16,
        rope_head_dim=8,
        v_head_dim=32,
        max_seq_len=128,
        dropout=0.0,
        norm_eps=1e-5,
        vocab_size=4096  # [修复] 必须提供 vocab_size，尽管在此测试中未被使用
    )

    device = "cpu"
    mla = MultiHeadLatentAttention(args).to(device).eval()

    rope_config = RoPEConfig(head_dim=args.rope_head_dim, max_seq_len=args.max_seq_len)
    rope = RoPE(rope_config).to(device)

    # 2. 模拟输入数据
    batch_size = 1
    history_len = 10

    # 历史数据 (History)
    x_history = torch.randn(batch_size, history_len, args.dim, device=device)
    # 当前数据 (Current Token)
    x_current = torch.randn(batch_size, 1, args.dim, device=device)

    # 3. 运行 Naive Forward (模拟训练模式，一次性输入全部序列)
    # 拼接 history 和 current
    x_full = torch.cat([x_history, x_current], dim=1)

    with torch.no_grad():
        output_naive = mla(x_full, rope, layer_idx=0)
        # 我们只关心最后一个 token 的输出
        last_token_naive = output_naive[:, -1:, :]

    print("✅ Naive forward pass completed.")

    # 4. 运行 Optimized Inference (模拟 KV Cache 模式)
    kv_cache = LatentKVCache(
        max_batch_size=batch_size,
        max_seq_len=args.max_seq_len,
        n_layers=1,
        kv_lora_rank=args.kv_lora_rank,
        rope_head_dim=args.rope_head_dim,
        device=device,
        dtype=torch.float32
    )

    with torch.no_grad():
        # Phase 1: Prefill (处理 History)
        # 为了测试严谨性，我们这里用循环模拟逐步生成，或者直接 Hack 进 Cache
        # 这里我们手动调用 inference 模式处理 history (逐个 token)
        # 在实际中 Prefill 通常走 Naive 模式生成 Cache，这里为了测试 forward_optimized，我们逐个塞进去

        for i in range(history_len):
            token = x_history[:, i:i + 1, :]
            _ = mla(token, rope, layer_idx=0, kv_cache=kv_cache, start_pos=i)

        # Phase 2: Decode (处理 Current Token)
        # 这是我们要验证的关键步骤
        last_token_opt = mla(x_current, rope, layer_idx=0, kv_cache=kv_cache, start_pos=history_len)

    print("✅ Optimized inference pass completed.")

    # 5. 对比结果
    print("\n📊 Results Comparison:")
    print(f"Naive Output Shape: {last_token_naive.shape}")
    print(f"Optimized Output Shape: {last_token_opt.shape}")

    diff = (last_token_naive - last_token_opt).abs().max().item()
    print(f"Max Difference: {diff:.8f}")

    if diff < 1e-4:
        print("\n🎉 SUCCESS: Optimized MLA implementation matches Naive implementation!")
    else:
        print("\n❌ FAILURE: Outputs do not match.")


if __name__ == "__main__":
    test_mla_correctness()
# END OF FILE: tests/test_mla_optimization.py