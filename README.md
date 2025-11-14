span

# LLM 从零到一终极教程 (LLM-From-Scratch-Ultimate-Tutorial)

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

在算法日益黑箱化、高级API唾手可得的时代，我们是否还记得那些让现代AI成为可能的基石？本项目旨在逆流而上，打造一个**终极的、体系化的、从零开始**的大语言模型（LLM）学习教程。

我们的目标，不是简单地调用`transformers`库，而是以一份详尽的技术需求清单为蓝图，带领学习者穿越理论的迷雾，**亲手用纯PyTorch实现每一个核心组件**，在消费级硬件（甚至纯CPU）上构建并预训练一个完整的、现代化的Transformer模型。

我们相信，唯有深入细节，方能洞见全局。本项目面向所有希望深度理解大模型底层原理的学生、研究者和工程师。

## ✨ 项目亮点

* **🧠 全流程覆盖**: 从数据处理、分词器训练，到模型定义、准工业级训练框架搭建，覆盖LLM预训练生命周期的每一个环节。
* **🔧 深度理论解析**: 为每一个核心模块（Attention, RoPE, RMSNorm...）都配有专属的`docs/*.md`文档，深入剖析其背后的数学原理与设计哲学。
* **💻 纯PyTorch手写实现**: 拒绝黑箱！所有模型组件均由基础PyTorch手写而成，最大化学习效果，让你真正理解每一行代码的意义。
* **🚀 准工业级训练框架**: 我们不止步于“能跑”，更追求“跑得好”。框架集成了**WandB/SwanLab双日志后端、断点续训、梯度累积、混合精度、学习率调度、内部激活值与梯度监控**等工业级训练必备功能。
* **📦 高效数据处理**: 实现了`Packed Sequences`等高级数据加载技术，并采用多进程优化数据预处理流程，确保在训练过程中每一份计算资源都用在刀刃上。
* **🧩 高度模块化与解耦**: 采用`config.yaml`统一管理配置，代码结构清晰，职责分离，极易扩展和二次开发。

## 🚀 快速开始

1. **环境配置**:
   ```bash
   uv pip install -r requirements.txt
   ```
2. **准备数据 (全自动流程)**:
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
3. **开始训练!**:
   ```bash
   # (可选) 在新终端启动 SwanLab 监控: swanlab watch
   # 运行冒烟测试
   python pretrain/train.py --config_path configs/pretrain_config.yaml
   ```

---

## 🗺️ 项目技术栈清单 (Technical Stack Checklist)

这是一个动态更新的清单，展示了本项目**已实现 `[x]`** 和**计划实现 `[ ]`** 的技术点。

### **一、数据处理流程**

- [X]  **数据收集**: 从开源数据集 (Hugging Face)
- [ ]  **数据清洗**:
  - [ ]  去重 (MinHash/SimHash)
  - [X]  编码规范化 (UTF-8，隐式实现)
  - [ ]  噪声过滤 (HTML标签等)
  - [ ]  质量过滤 (语言检测, 困惑度过滤, PII)
- [ ]  **数据分析**: 词频、文档长度分布等 (提供占位脚本)
- [X]  **数据格式化**:
  - [X]  **Packed Sequences** (打包序列，提高训练效率)
  - [ ]  文档分块策略

### **二、Tokenizer构建**

- [X]  **分词算法**:
  - [X]  **BPE** (Byte Pair Encoding)
  - [ ]  WordPiece
  - [ ]  Unigram
- [X]  **词表构建**:
  - [X]  词表大小选择
  - [X]  自定义特殊Token (`<|endoftext|>`, `<|pad|>`, etc.)
- [X]  **实现细节**:
  - [X]  [HuggingFace Tokenizers] 高性能训练 (`train_tokenizer.py`)
  - [X]  [教学版] 纯Python手写BPE (`bpe_teach.py`)

### **三、模型架构设计**

- [X]  **基础架构**: Transformer (Decoder-only)
- [ ]  **其他架构**: State Space Models (Mamba), RWKV, Hybrid

- **注意力机制 (Attention)**:
  - [X]  **标准变体**: MHA, MQA, GQA
  - [ ]  **稀疏注意力**:
    - [ ]  Sliding Window (Longformer)
    - [ ]  NSA (Native Sparse Attention)
    - [ ]  DSA (DeepSeek Sparse Attention)
    - [ ]  MoBA (Mixture of Block Attention - Kimi)
    - [ ]  KDA (Kimi Delta Attention)
  - [ ]  **线性/次线性注意力**:
    - [ ]  FLASH
    - [ ]  Performer
    - [ ]  1-Liner Attention
    - [ ]  Lightning Attention (MiniMax)
    - [ ]  Logic Attention
  - [ ]  **压缩KV**: MLA (Multi-head Latent Attention - DeepSeek)
- **前馈网络 (FFN)**:
  - [X]  **GLU变体**: SwiGLU
  - [ ]  **稀疏FFN**: MoE (Mixture of Experts)
- **激活函数**:
  - [X]  SiLU/Swish (隐式在SwiGLU中)
  - [ ]  GELU
- **位置编码 (Positional Encoding)**:
  - [X]  **绝对位置编码**: Learned, Sinusoidal
  - [X]  **相对位置编码**: RoPE, ALiBi (以辅助函数形式实现)
  - [ ]  **长度外推**: YaRN, PI
- **归一化层 (Normalization)**:
  - [X]  **LayerNorm** (Pre-LN架构中)
  - [X]  **RMSNorm**
  - [X]  **BatchNorm** (用于教学对比)
  - [X]  **Qwen2RMSNorm** (`1+w` 技巧)
- **残差连接**:
  - [X]  **Pre-LN** 架构

### **四、训练流程**

- [X]  **权重初始化**:
  - [X]  GPT-2风格标准初始化
  - [ ]  Kaiming/Xavier

- **预训练 (Pre-training)**:
  - [X]  **训练目标**: CLM (Causal Language Modeling)
  - [X]  **超参数配置**: 通过 `config.yaml` 统一管理
  - [X]  **优化器**:
    - [ ]  发展历程: 最小二乘法 -> BGD -> **SGD**
    - [X]  **AdamW** (带权重衰减分离)
    - [ ]  **Muon**
    - [ ]  Adafactor, Lion
  - [X]  **混合精度训练**:
    - [X]  CPU `bfloat16`
    - [X]  GPU `float16` (已支持`GradScaler`)
  - [ ]  **分布式训练**: DDP, FSDP, DeepSpeed ZeRO
- **训练优化技术**:
  - [ ]  **Flash Attention**
  - [X]  **Packed Sequences**
- **Checkpoint管理**:
  - [X]  **完整状态保存**: 模型, 优化器, 调度器, `GradScaler`
  - [X]  **断点续训** (latest/best)
  - [X]  最佳模型保存

### **五、后训练 (Post-training)**

- [ ]  **监督微调 (SFT)**:
  - [ ]  SFT数据准备与训练脚本 (`sft_train.py` 为占位符)
  - [ ]  **PEFT技术**:
    - [ ]  LoRA
    - [ ]  QLoRA
    - [ ]  **进阶变体**: AdaLoRA, DoRA, PiSSA
- [ ]  **强化学习对齐**:
  - [ ]  **奖励模型 (RM)** 训练 (`reward_model.py` 为占位符)
  - [ ]  **RL算法**:
    - [ ]  **经典**: PPO, TRPO
    - [ ]  **离线/简化**: DPO, ORPO, IPO, KTO
    - [ ]  **进阶**: GRPO, GSPO, GMPO, DAPO, BAPO

### **六、模型评估**

- **自动评估指标**:
  - [X]  **困惑度 (Perplexity)**
  - [ ]  **生成质量**: BLEU, ROUGE
  - [ ]  **代码能力**: pass@k
- **Benchmark评估**:
  - [ ]  **通用能力**: MMLU, C-Eval
  - [ ]  **推理能力**: GSM8K, MATH
  - [ ]  **代码能力**: HumanEval, MBPP
  - [ ]  **长文本能力**: LongBench, Needle In A Haystack
- **训练过程监控**:
  - [X]  **基础指标**: Loss, Learning Rate
  - [X]  **内部指标 (遥测)**: 通过Hooks监控激活值范数, 梯度范数

### **七、模型部署与推理优化**

- [ ]  **采样策略**:
  - [ ]  Greedy, Top-K, Top-P, Temperature (`sampling.py` 为占位符)
- [ ]  **推理加速**:
  - [ ]  **KV Cache** (`kv_cache.py` 为占位符)
  - [ ]  Speculative Decoding
- [ ]  **模型压缩**:
  - [ ]  **量化**: GPTQ, AWQ (`quantization.py` 为占位符)
  - [ ]  **剪枝**: 结构化与非结构化 (`pruning.py` 为占位符)
  - [ ]  **知识蒸馏**

## ❤️ 欢迎贡献

本项目是一个开放的、持续生长的学习资源。我们深知其中必有不足与疏漏，**我们热切地欢迎任何形式的贡献**！

无论是一个错字的修正、一行注释的补充、一个Bug的修复，还是一个新功能的PR，都是对开源社区的宝贵贡献。请不要犹豫，Fork本项目并发起你的Pull Request吧！

---

<div align="center">
  <samp>追风赶月莫停留，平芜尽处是春山。</samp>
  <br>
  <samp>Chasing the wind and the moon, we shall not stay; where the plains end, the verdant mountains of spring await.</samp>
</div>
