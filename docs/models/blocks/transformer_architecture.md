# 理论解析: Transformer 整体架构

我们已经实现了构成Transformer的各个核心组件。现在，是时候将它们组装起来，理解一个完整的Transformer语言模型是如何构建的了。

## 1. Transformer Block: 模型的基石

Transformer模型的主体是由N个完全相同的**Transformer Block**堆叠而成的。我们采用的是**Pre-Normalization (Pre-LN)**架构，这种架构被证明比原版的Post-LN架构训练起来更稳定，并被GPT-2、LLaMA等现代模型广泛采用。

### Pre-LN Transformer Block 的数据流

一个输入张量 `x` 在一个Block中的完整旅程如下：

```
      x (来自上一层)
      |
      |-----> Pre-Normalization (RMSNorm) -> Multi-Head Attention ----->(+)---> h
      |                                                                 |
      +-----------------------------------------------------------------+ (残差连接 1)
                                                                        |
      |-----> Pre-Normalization (RMSNorm) -> Feed-Forward Network ----->(+)---> out
      |                                                                 |
      +-----------------------------------------------------------------+ (残差连接 2)
                                                                        |
                                                                   (输出到下一层)
```

**伪代码表示:**
```python
def transformer_block(x, attention_module, ffn_module, norm1, norm2):
    # 第一个子层：多头注意力
    residual_1 = x
    x_normalized_1 = norm1(x)
    attention_output = attention_module(x_normalized_1)
    h = residual_1 + attention_output # 残差连接 1

    # 第二个子层：前馈网络
    residual_2 = h
    h_normalized_2 = norm2(h)
    ffn_output = ffn_module(h_normalized_2)
    out = residual_2 + ffn_output # 残差连接 2
    
    return out
```

这种“先归一化，再计算，最后残差连接”的模式，保证了每一层的输入分布都是稳定的，从而使得梯度能够更平滑地传播，尤其是在非常深的网络中。

## 2. 完整的Transformer语言模型

将多个Transformer Block堆叠起来，并在首尾加上必要的层，就构成了一个完整的语言模型。

### 宏观架构

```
      Input Token IDs (e.g.,)
             |
             v
      Token Embedding Layer  (将ID转换为向量)
             |
             +------------- Positional Encoding (RoPE, 作用于Attention内部的Q/K)
             |
             v
      Transformer Block 1
             |
             v
      Transformer Block 2
             |
             v
            ... (重复 N 次)
             |
             v
      Transformer Block N
             |
             v
      Final Normalization Layer (RMSNorm)
             |
             v
      LM Head (线性层，将向量映射回词表大小)
             |
             v
      Output Logits (每个位置上所有词的概率分布)
```

### 关键组件解释

-   **Token Embedding (`nn.Embedding`)**: 一个可学习的查找表，将输入的离散Token ID映射为高维的、密集的向量表示。
-   **Positional Encoding (RoPE)**: 我们在模型顶层实例化一个RoPE模块。在`forward`过程中，这个模块的实例会被传递给每一个`TransformerBlock`，然后在`Attention`层内部被调用，作用于Q和K向量，从而注入位置信息。
-   **N x Transformer Blocks (`nn.ModuleList`)**: 模型的核心，负责对输入的向量序列进行深度特征提取和上下文理解。
-   **Final Normalization Layer**: 在送入最后的输出层之前，对所有特征进行最后一次归一化，以稳定输出。
-   **LM Head (`nn.Linear`)**: 一个线性层，它的权重通常与Token Embedding层共享（一种称为“权重绑定”的技术，可以减少参数量并提高性能）。它的作用是将Transformer最终输出的高维向量，投影回词汇表的大小，得到每个位置上每个词的“得分”（Logits）。对Logits应用Softmax函数，就可以得到下一个词的概率分布。

通过这个流程，我们就从一个简单的Token ID序列，得到了一个包含了丰富上下文信息、可用于预测下一个词的概率分布。这就是Transformer语言模型的核心工作原理。
