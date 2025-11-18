# FILE: models/blocks/attention/linear.py
"""
【流派二：线性注意力】
包含：
1. LinearAttention: 基于 ReLU 核函数的 O(N) 复杂度注意力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    基于 Kernel Trick 的线性注意力。
    Attn(Q, K, V) = (phi(Q) @ phi(K)^T) @ V
    """

    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def feature_map(self, x):
        # Simple ReLU feature map as used in "Transformers are RNNs"
        return F.relu(x)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        if paged_attention_inputs is not None:
            # 线性注意力的 Paged 实现需要维护 RNN 状态而非 KV Blocks，
            # 这是一个完全不同的工程实现，目前暂不支持。
            raise NotImplementedError("PagedAttention not yet supported for Linear Attention variant.")

        bs, seq_len, _ = x.shape

        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = rope.apply_rotary_emb(q)
        k = rope.apply_rotary_emb(k)

        Q = self.feature_map(q)
        K = self.feature_map(k)

        # Standard Causal Linear Attention (Naive Implementation)
        # 为了保持因果性，这里使用了 Masked Matmul，复杂度退化回 O(N^2)。
        # 真正的 O(N) 训练需要使用累积求和技巧 (Cumulative Sum) 或自定义 CUDA Kernel。
        # 但为了代码的可读性和纯 PyTorch 实现，我们展示逻辑正确性优先。

        attn = torch.matmul(Q, K.transpose(2, 3))
        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            attn = attn.masked_fill(self.mask[:, :, start_pos:current_seq_len, :current_seq_len] == float("-inf"), 0)

        denom = attn.sum(dim=-1, keepdim=True) + 1e-5
        attn = attn / denom

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.resid_dropout(self.wo(output))

# END OF FILE: models/blocks/attention/linear.py