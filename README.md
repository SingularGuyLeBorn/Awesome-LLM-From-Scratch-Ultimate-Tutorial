<div align="center">
<span style="font-size: 48px; font-weight: bold; ">LLM 从零到一终极教程</span>
<br>
<span style="font-size: 20px;">(LLM-From-Scratch-Ultimate-Tutorial)</span>

**一个史诗级的、从零手写大语言模型的终极指南，旨在征服LLM宇宙的每一个角落。**

</div>

<p align="center">
  <img alt="Author" src="https://img.shields.io/badge/Author-会长-blue.svg">
  <img alt="Technical Support" src="https://img.shields.io/badge/AI_Support-Gemini_Pro-purple.svg">
  <img alt="Community" src="https://img.shields.io/badge/Community-DeepWiki-orange.svg">
  <br>
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python">
  <img alt="Framework" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch">
  <img alt="Package Manager" src="https://img.shields.io/badge/uv-0.1+-007ACC">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
  <img alt="Contributions" src="https://img.shields.io/badge/PRs-Welcome-ff69b4.svg">
</p>

---

## 📖 项目愿景

在算法日益黑箱化、高级API唾手可得的时代，我们是否还记得那些让现代AI成为可能的基石？许多教程要么止步于理论，要么过度依赖 `transformers` 等高级库，让学习者在抽象的API调用中“知其然，而不知其所以然”。

本项目旨在打破这一困境，打造一个**终极的、体系化的、从零开始**的大语言模型（LLM）学习与实践平台。我们的核心哲学是 **“代码即理论，实践出真知”**。

**我们不满足于简单调用，我们追求深度理解。** 我们将以一份详尽的技术蓝图为指引，带领你穿越理论的迷雾，**亲手用纯PyTorch实现LLM生命周期的每一个核心环节**：从数据处理、分词器构建，到模型架构（Attention, RoPE, SwiGLU...）、准工业级训练框架，再到监督微调（SFT）与强化学习对齐（RLHF），最终实现高效推理与交互式应用。

本项目面向所有希望深度理解大模型底层原理的学生、研究者和工程师。无论你的目标是学术研究、工程应用还是技术面试，这里都将为你提供最坚实的基础。

---

## ✨ 项目亮点

本项目在设计上严格遵循三大原则：**理论清晰、性能高效、易于上手**。

#### 1. 🎯 **面向实战的全生命周期覆盖 (Full Lifecycle)**

我们不仅教你如何构建一个`Transformer`，更带你走完从原始数据到可对话模型的完整工业流程。

*   **数据工程**: 全自动化的数据流水线，涵盖下载、多进程预处理、高效Token化及二进制文件构建 (`Packed Sequences`)。
*   **从零预训练 (Pre-training)**: 搭建准工业级训练框架，支持混合精度、动态梯度裁剪、多后端日志（WandB/SwanLab）和断点续训，在CPU上即可完整运行。
*   **对齐微调 (Alignment)**: 实现从监督微调（SFT）、奖励模型（RM）训练，到高级强化学习对齐（DPO, PPO, GRPO/GSPO）的全套方案。
*   **高效推理 (Inference)**: 手写实现KV缓存与 **PagedAttention**，模拟 vLLM 的内存管理机制，并提供 OpenAI 兼容的 API Server。

#### 2. 🔧 **“准工业级”代码实现 (Quasi-Industrial Grade)**

在保持教学清晰性的前提下，我们追求代码的性能与健壮性，确保项目不仅能跑，而且跑得好，并能轻松扩展。

*   **CPU友好，GPU就绪**: 所有核心流程均可在消费级CPU（如16GB内存）上流畅运行。代码遵循PyTorch最佳实践，无缝迁移至GPU环境即可获得数十倍加速。
*   **性能优化**: 在关键路径（如数据加载、位置编码）采用工业级优化方案，拒绝教学代码的性能瓶颈。
*   **高度模块化与解耦**: 采用`config.yaml`统一管理所有实验参数，代码结构清晰，职责分离（数据、模型、训练、推理），极易扩展和二次开发。
*   **深度监控**: 通过PyTorch Hooks实现对模型内部激活值与梯度的“遥测”，让你像医生一样诊断模型的健康状况。

#### 3. 🎓 **专为学习者设计的低门槛 (Learner-Friendly)**

我们深知学习曲线的陡峭，因此在每个环节都力求降低入门门槛。

*   **纯PyTorch实现**: 拒绝黑箱！所有核心算法（Attention, RoPE, RMSNorm, AdamW, DPO...）均由基础PyTorch手写而成，最大化学习效果，让你真正理解每一行代码的意义。
*   **一键启动**: 提供完整的自动化脚本，只需配置好环境，即可一键完成数据准备到模型训练的全过程。
*   **数学要求低**: 理解本项目核心代码仅需本科水平的线性代数基础。我们特意选用了理论更直观、实现更简洁的算法（如RoPE），并为你准备了详尽的配套理论文档。
*   **深度理论文档**: 为每一个核心模块都配有专属的`docs/*.md`文档，深入浅出地剖析其背后的数学原理与设计哲学，将理论与代码完美对应。

## 🚀 快速开始

1.  **环境配置**:
    ```bash
    uv pip install -r requirements.txt
    ```
2.  **准备数据 (全自动流程)**:
    ```bash
    # 下载原始数据 (TinyStories)
    python data_pipeline/download/download_tinystories.py
    # 训练分词器 (使用HuggingFace Tokenizers, 速度快)
    python data_pipeline/tokenizer/train_tokenizer.py --vocab_size 4096 --data_limit_mb 100
    # 编码为中间格式 (多进程)
    python data_pipeline/processing/encode_stories.py
    # 构建最终 .bin 训练文件
    python data_pipeline/processing/build_pretrain_bins.py
    ```
3.  **开始训练!**:
    ```bash
    # (可选) 在新终端启动 SwanLab 监控: swanlab watch
    # 运行冒烟测试
    python pretrain/train.py --config_path configs/pretrain/1.4M_pretrain_fast.yaml
    ```
4.  **与你的模型聊天!**:
    ```bash
    # 训练完成后，找到你的最佳检查点路径
    # 加上 --quantize 参数以在 CPU 上使用 int8 量化加速
    python inference/chat.py --checkpoint_path [你的最佳模型ckpt路径] --quantize
    ```

---

## 🗺️ 项目技术栈清单 (Technical Stack Checklist)

这是一个动态更新的清单，展示了本项目**已实现 `[x]`** 和**计划实现 `[ ]`** 的技术点。

### **一、数据处理流程**

-   [X]  **数据收集**: 从开源数据集 (Hugging Face)
-   [ ]  **数据清洗**:
    -   [ ]  去重 (MinHash/SimHash)
    -   [X]  编码规范化 (UTF-8，隐式实现)
    -   [ ]  噪声过滤 (HTML标签等)
    -   [ ]  质量过滤 (语言检测, 困惑度过滤, PII)
-   [ ]  **数据分析**: 词频、文档长度分布等 (提供占位脚本)
-   [X]  **数据格式化**:
    -   [X]  **Packed Sequences** (打包序列，提高训练效率)
    -   [ ]  文档分块策略

### **二、Tokenizer构建**

-   [X]  **分词算法**:
    -   [X]  **BPE** (Byte Pair Encoding)
    -   [ ]  WordPiece
    -   [ ]  Unigram
-   [X]  **词表构建**:
    -   [X]  词表大小选择
    -   [X]  自定义特殊Token (`<|endoftext|>`, `<|pad|>`, etc.)
-   [X]  **实现细节**:
    -   [X]  [HuggingFace Tokenizers] 高性能训练 (`train_tokenizer.py`)
    -   [X]  [教学版] 纯Python手写BPE (`bpe_teach.py`)

### **三、模型架构设计**

-   [X]  **基础架构**: Transformer (Decoder-only)
-   [ ]  **其他架构**: State Space Models (Mamba), RWKV, Hybrid

-   **注意力机制 (Attention)**:
    -   [X]  **标准变体**: MHA, MQA, GQA
    -   [ ]  **稀疏注意力**:
        -   [ ]  Sliding Window (Longformer)
        -   [ ]  NSA (Native Sparse Attention)
        -   [ ]  DSA (DeepSeek Sparse Attention)
    -   [ ]  **线性/次线性注意力**:
        -   [ ]  FLASH
        -   [ ]  Lightning Attention
    -   [ ]  **压缩KV**: MLA (Multi-head Latent Attention - DeepSeek)
-   **前馈网络 (FFN)**:
    -   [X]  **GLU变体**: SwiGLU
    -   [ ]  **稀疏FFN**: MoE (Mixture of Experts)
-   **激活函数**:
    -   [X]  SiLU/Swish (隐式在SwiGLU中)
    -   [ ]  GELU
-   **位置编码 (Positional Encoding)**:
    -   [X]  **绝对位置编码**: Learned, Sinusoidal
    -   [X]  **相对位置编码**: RoPE (含 Paged 支持), ALiBi
    -   [ ]  **长度外推**: YaRN
-   **归一化层 (Normalization)**:
    -   [X]  **LayerNorm**
    -   [X]  **RMSNorm**
    -   [X]  **Qwen2RMSNorm** (`1+w` 技巧)
-   **残差连接**:
    -   [X]  **Pre-LN** 架构

### **四、训练流程**

-   [X]  **权重初始化**: GPT-2风格标准初始化
-   **预训练 (Pre-training)**:
    -   [X]  **训练目标**: CLM (Causal Language Modeling)
    -   [X]  **优化器**: AdamW
    -   [X]  **混合精度训练**: CPU `bfloat16`, GPU `float16` (`GradScaler`)
    -   [X]  **分布式训练**: DDP (Distributed Data Parallel)
-   **Checkpoint管理**:
    -   [X]  **完整状态保存**: 模型, 优化器, 调度器, `GradScaler`
    -   [X]  **断点续训**

### **五、后训练 (Post-training)**

-   [X]  **监督微调 (SFT)**: 全量微调, LoRA
-   [X]  **强化学习对齐**:
    -   [X]  **奖励模型 (RM)**
    -   [X]  **RL算法**: PPO, DPO, ORPO, GRPO, GSPO

### **六、模型评估**

-   **自动评估指标**:
    -   [X]  **困惑度 (Perplexity)**
    -   [ ]  **Benchmark评估**: MMLU, GSM8K (计划中)
-   **训练过程监控**:
    -   [X]  **基础指标**: Loss, LR
    -   [X]  **内部指标 (遥测)**: 激活值范数, 梯度范数

### **七、模型部署与推理优化**

-   [X]  **采样策略**: Greedy, Top-K, Top-P, Temperature
-   [X]  **推理加速**:
    -   [X]  **KV Cache**
    -   [X]  **PagedAttention** (Mini-vLLM 实现)
-   [X]  **服务化**: OpenAI 兼容 API Server
-   [X]  **模型压缩**:
    -   [X]  **量化**: 动态 Int8 量化 (CPU Friendly)
    -   [ ]  **剪枝**

## ❤️ 欢迎贡献

本项目是一个开放的、持续生长的学习资源。我们深知其中必有不足与疏漏，**我们热切地欢迎任何形式的贡献**！

无论是一个错字的修正、一行注释的补充、一个Bug的修复，还是一个新功能的PR，都是对开源社区的宝贵贡献。请不要犹豫，Fork本项目并发起你的Pull Request吧！

---

<div align="center">
  <samp>追风赶月莫停留，平芜尽处是春山。</samp>
  <br>
  <samp>Chasing the wind and the moon, we shall not stay; where the plains end, the verdant mountains of spring await.</samp>
</div>