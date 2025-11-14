# FILE: docs/models/blocks/attention/attention.md
# 理论解析: 注意力机制全景 (MHA, MQA, GQA)

## 1. 注意力机制的核心思想

自注意力（Self-Attention）是Transformer模型的基石。其核心思想是为序列中的每个Token计算一个“注意力分数”分布，这个分数决定了在生成该Token的表示时，应该对序列中其他所有Token“关注”多少。

**数学公式 (Scaled Dot-Product Attention)**:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
-   **Q (Query)**: 代表当前Token，它要去“查询”别的Token。
-   **K (Key)**: 代表序列中被查询的Token，与Query计算相似度。
-   **V (Value)**: 代表序列中被查询的Token的实际内容，根据相似度进行加权求和。
-   **√d_k**: 缩放因子，用于稳定梯度。

---

## 2. 主流注意力变体 (MHA, MQA, GQA)

这些变体的核心区别在于**Key和Value头的数量**，这直接影响了模型的**参数量**和推理时的**KV缓存大小**，对推理速度至关重要。

### 2.1 Multi-Head Attention (MHA)
-   **思想**: “兼听则明”。将`d_model`维空间拆分为`h`个独立的“头”，每个头学习不同的注意力模式。每个Query头都有自己独立的Key和Value头。
-   **结构**: `h`个Query头, `h`个Key头, `h`个Value头。
-   **KV缓存**: `2 * h * d_head * seq_len` = `2 * d_model * seq_len`。这是推理速度的主要瓶颈。

### 2.2 Multi-Query Attention (MQA)
-   **思想**: “一问多答”。所有`h`个Query头共享**唯一**的一组Key和Value头。
-   **结构**: `h`个Query头, **1**个Key头, **1**个Value头。
-   **KV缓存**: `2 * d_head * seq_len`。相比MHA，缓存大小减少了`h`倍，极大提升了推理速度。
-   **缺点**: 可能会因所有头共享K/V而导致一定的性能损失。

### 2.3 Grouped-Query Attention (GQA)
-   **思想**: “分组讨论”，是MHA和MQA的折中。将`h`个Query头分成`g`组，组内共享同一组Key和Value头。
-   **结构**: `h`个Query头, `g`个Key头, `g`个Value头 (1 < g < h)。
-   **KV缓存**: `2 * g * d_head * seq_len`。相比MHA，缓存大小减少了`h/g`倍。
-   **优点**: 在保持接近MHA性能的同时，大幅减少了KV缓存，实现了速度和质量的平衡。

| 方案 | Query头数 | KV头数 | 参数优势 (vs MHA) | KV缓存优势 (vs MHA) | 代表模型 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MHA** | `h` | `h` | 1x | 1x | BERT, GPT-3 |
| **GQA** | `h` | `g` | ≈ **h/g 倍** | **h/g 倍** | LLaMA-2/3, Mixtral |
| **MQA** | `h` | **1** | ≈ **h 倍** | **h 倍** | PaLM, Falcon |

---

## 3. 前沿探索: Multi-head Latent Attention (MLA)

MLA是DeepSeek-V2模型提出的，旨在从根本上解决KV缓存与序列长度`seq_len`线性相关的问题。

-   **思想**: “读书先看目录”。不再让Query直接关注整个冗长的上下文(K,V)，而是先将K,V**压缩**成一个短小的、固定长度的“摘要”或“潜序列”(Latent Sequence)，然后Query只与这个摘要进行交互。
-   **优点**: KV缓存大小与序列长度**完全无关**！这对于无限长上下文是革命性的。
-   **缺点**: 压缩过程可能损失信息，对压缩网络的设计要求高。

## 4. 参考文献
-   [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
-   [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
-   [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - MLA
# END OF FILE: docs/models/blocks/attention/attention.md