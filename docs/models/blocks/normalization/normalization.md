# 理论解析: 归一化层全景 (BN, LN, RMSNorm, Qwen2RMSNorm)

## 1. 为什么需要归一化？

在深度神经网络中，每一层的输出都会成为下一层的输入。在训练过程中，由于前一层参数的不断更新，后一层的输入数据分布会持续发生变化，这种现象被称为**内部协变量偏移 (Internal Covariate Shift)**。这种偏移会迫使模型不断适应变化的输入分布，降低学习效率，并可能导致梯度消失或爆炸。

归一化层通过在网络中间层强制执行固定的数据分布，极大地稳定了训练过程。

---

## 2. BatchNorm (BN): 计算机视觉的王者

BatchNorm 在CV领域取得了巨大成功，它的核心思想是在**批次（Batch）维度**上对数据进行归一化。

### 数学推导
对于一个mini-batch的输入 `x` (形状 `B, L, D`)，BN对**每一个特征维度 `d`**，在所有样本 `B` 和所有位置 `L` 上计算均值和方差。

1.  **训练阶段**:
    -   计算均值: $ \mu_d = \frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{l=1}^{L} x_{b, l, d} $
    -   计算方差: $ \sigma_d^2 = \frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{l=1}^{L} (x_{b, l, d} - \mu_d)^2 $
    -   归一化: $ \hat{x}_{b, l, d} = \frac{x_{b, l, d} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} $
    -   仿射变换: $ y_{b, l, d} = \gamma_d \cdot \hat{x}_{b, l, d} + \beta_d $

2.  **推理阶段**:
    -   使用训练期间通过指数移动平均累积的全局统计量 `running_mean` 和 `running_var` 进行归一化。

### 为什么BN不适用于LLM？
1.  **对Batch Size敏感**: BN的性能严重依赖于足够大的Batch Size来估算准确的全局统计量。在LLM中，由于显存限制，Batch Size通常很小，这使得BN的统计量充满噪声。
2.  **序列长度不一致**: 在NLP任务中，一个batch内的序列长度往往不同（需要padding），BN的统计方式会受到padding的影响。
3.  **训练与推理不一致**: 训练时和推理时使用不同的统计量，给模型部署和量化带来了复杂性。

---

## 3. LayerNorm (LN): Transformer的经典选择

为了解决BN的问题，LayerNorm被提出。它完全在**单个样本内部**进行归一化，与Batch Size无关。

### 数学推导
对于单个输入样本 `x` (形状 `D`)，LN在**特征维度 `D`** 上计算均值和方差。

1.  **计算均值**: $ \mu = \frac{1}{D} \sum_{i=1}^{D} x_i $
2.  **计算方差**: $ \sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^2 $
3.  **归一化**: $ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $
4.  **仿射变换**: $ y_i = \gamma \cdot \hat{x}_i + \beta $ (注意 `γ` 和 `β` 是维度为 `D` 的向量)

LN完美地解决了BN在LLM中的所有问题，因此成为原版Transformer的标配。

---

## 4. RMSNorm: 现代LLM的主流

后续研究发现，LN中的**中心化操作（减去均值）**对性能的贡献远小于**缩放操作（除以方差）**。RMSNorm (Root Mean Square Normalization) 正是基于这一观察的简化。

### 数学推导
RMSNorm移除了均值计算，直接通过输入的**均方根**进行缩放。

1.  **计算均方根 (RMS)**: $ \text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2} $
2.  **归一化**: $ \hat{x}_i = \frac{x_i}{\sqrt{\text{RMS}(x)^2 + \epsilon}} = \frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D} x_j^2 + \epsilon}} $
3.  **仿射变换**: $ y_i = g_i \cdot \hat{x}_i $ (只有一个可学习的增益参数 `g`)

### 为什么RMSNorm成为主流？
-   **vs BatchNorm**: 继承了LN的优点，完全与Batch Size无关，非常适合LLM。
-   **vs LayerNorm**:
    -   **计算高效**: 省去了均值和方差的计算，直接计算均方根，在GPU上能节省约7%-64%的计算时间。
    -   **性能相当甚至更优**: 大量实验（如LLaMA）证明，这种简化不仅没有损害模型性能，有时甚至有微小提升。这表明对于Transformer来说，重新缩放特征的幅度是归一化中最重要的部分。

---

## 5. Qwen2RMSNorm: `(1+w)`的炼丹技巧

`Qwen3-next` (以及Qwen2) 中的实现，是对RMSNorm的一个精妙的**重参数化 (re-parameterization)**。

### 理论分析
标准RMSNorm的公式是 $ y = g \cdot \hat{x} $，其中 `g` 通常初始化为**全1**的向量。这意味着在训练开始时，这一层会对输入进行非平凡的缩放。

Qwen2RMSNorm的公式是 $ y = (1+w) \cdot \hat{x} $，其中 `w` 初始化为**全0**的向量。

这带来了什么好处？
-   **初始化时的恒等映射**: 在训练刚开始时（t=0），`w` 几乎为0，所以缩放因子 `(1+w)` 几乎为1。这意味着整个Qwen2RMSNorm层在初始化时**近似于一个恒等函数**（只做归一化，不做缩放）。
-   **更稳定的训练开局**: 让网络层在初始化时更接近恒等映射，是一种被证明可以有效稳定深度网络训练的技巧。它为梯度提供了一条更“平滑”的路径，避免了在训练初期由于随机初始化的巨大参数导致输出发生剧烈变化。模型可以在此基础上更平稳地学习需要进行的缩放调整。

这是一个典型的“用一个好的归纳偏置来启动训练”的例子，虽然改动极小，但对超大模型的训练稳定性和最终性能可能有正面影响。

### 代码实现
```python
class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 关键：权重初始化为0
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 关键：使用 (1 + weight) 作为缩放因子
        return (1 + self.weight) * normalized_x
```

---

## 6. 总结对比

| 特性 | BatchNorm | LayerNorm | RMSNorm | Qwen2RMSNorm |
| :--- | :--- | :--- | :--- | :--- |
| **归一化维度** | 批次 | 特征 | 特征 | 特征 |
| **核心操作** | 中心化+缩放 | 中心化+缩放 | **仅缩放** | **仅缩放 (1+w)** |
| **对Batch敏感**| **是** | 否 | 否 | 否 |
| **计算复杂度** | 高 | 中 | **低** | **低** |
| **初始化行为** | 复杂 | 缩放+偏移 | 缩放 (`g=1`) | **近似恒等** (`1+w=1`)|
| **主要应用** | CV, MLP | 早期Transformer | **现代LLM (主流)** | **SOTA LLM (技巧)** |