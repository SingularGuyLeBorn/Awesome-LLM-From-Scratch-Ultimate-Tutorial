# FILE: models/blocks/attention/linear.py
"""
【流派二：线性注意力 - 真正的 O(N) 实现】
包含：
1. LinearAttention: 基于 ReLU 核函数和累积求和 (CumSum) 的 O(N) 复杂度注意力。

   Ref: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (Katharopoulos et al., 2020)

   公式重写:
   Attention(Q, K, V)_i = \frac{\phi(Q_i) \sum_{j=1}^i \phi(K_j)^T V_j}{\phi(Q_i) \sum_{j=1}^i \phi(K_j)^T}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    基于 Kernel Trick 的线性注意力。
    使用累积求和 (CumSum) 实现并行训练时的 O(N) 复杂度。
    """

    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # 线性注意力通常不需要太大的 head_dim，因为内存消耗是 d_head^2
        # 如果 args.dim 很大，建议增加 n_heads 来减小 head_dim

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)

        # 这里的 eps 用于分母防止除零
        self.eps = 1e-6

    def feature_map(self, x):
        """
        特征映射函数 phi(x)。
        原文使用 elu(x) + 1，这里使用更简单的 relu(x) 以保证非负性，
        这在 "Efficient Attention: Attention with Linear Complexities" 中也被广泛使用。
        """
        return F.relu(x)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        """
        Args:
            x: (batch_size, seq_len, dim)
        """
        if paged_attention_inputs is not None:
            raise NotImplementedError("PagedAttention not yet supported for Linear Attention variant.")

        bs, seq_len, dim = x.shape

        # 1. 投影 Q, K, V
        # Shape: (B, L, H, D)
        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bs, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(bs, seq_len, self.n_heads, self.head_dim)

        # 2. 应用 RoPE
        # 注意: 线性注意力应用 RoPE 存在理论上的争议，因为 RoPE 依赖于相对位置的旋转，
        # 而线性注意力的核函数分解破坏了这种旋转不变性。
        # 但在实践中，许多实现（如 TransnormerLLM）仍然加上 RoPE。
        # 这里我们为了保持接口一致性，加上 RoPE，需先转置为 (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = rope.apply_rotary_emb(q)
        k = rope.apply_rotary_emb(k)
        # 转置回 (B, L, H, D) 以便进行 cumsum 操作
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # 3. 应用特征映射 phi(x)
        # Q, K >= 0 对于数值稳定性至关重要
        Q = self.feature_map(q)
        K = self.feature_map(k)

        # 4. 真正的 O(N) 核心逻辑 (Parallel Scan / CumSum)

        # 4.1 计算 KV 状态矩阵
        # K: (B, L, H, D), V: (B, L, H, D)
        # 我们需要计算外积 K^T * V。
        # 这是一个 (D x D) 的矩阵。
        # 使用 einsum: 对于每个 batch, 每个 length, 每个 head: vector K * vector V -> matrix D*D
        # kv_state shape: (B, L, H, D, D)
        # 警告: 这会消耗大量显存 (L * D^2)。如果 head_dim=128, D^2=16384。
        kv_state = torch.einsum("blhd,bhlm->blhdm", K, v)

        # 4.2 计算累积和 (CumSum) 实现因果掩码
        # S_i = \sum_{j=1}^i K_j^T V_j
        kv_cumsum = torch.cumsum(kv_state, dim=1)

        # 4.3 计算分子的累积和 (Normalizer Z)
        # Z_i = \sum_{j=1}^i K_j
        # Shape: (B, L, H, D)
        k_cumsum = torch.cumsum(K, dim=1)

        # 5. 计算输出
        # Numerator = Q_i * S_i
        # Einsum: (B, L, H, D) * (B, L, H, D, M) -> (B, L, H, M)
        numerator = torch.einsum("blhd,blhdm->blhm", Q, kv_cumsum)

        # Denominator = Q_i * Z_i
        # Einsum: (B, L, H, D) * (B, L, H, D) -> (B, L, H)
        denominator = torch.einsum("blhd,blhd->blh", Q, k_cumsum)

        # 扩展分母维度以进行广播: (B, L, H) -> (B, L, H, 1)
        denominator = denominator.unsqueeze(-1)

        # 6. 归一化
        output = numerator / (denominator + self.eps)

        # 7. 输出投影
        # (B, L, H, D) -> (B, L, D)
        output = output.reshape(bs, seq_len, dim)
        return self.resid_dropout(self.wo(output))

# END OF FILE: models/blocks/attention/linear.py