<div align="center">

<img src="https://img.shields.io/badge/🧬-LLM_From_Scratch-black?style=for-the-badge&logo=github" alt="Logo" height="40">

<h1 style="font-size: 42px; margin-bottom: 0px;">LLM 从零到一终极教程</h1>
<span style="font-size: 18px; color: #666;">(LLM-From-Scratch-Ultimate-Tutorial)</span>

<p style="font-size: 16px; max-width: 800px; margin: 20px auto;">
<b>一个史诗级的、从零手写大语言模型的终极指南。</b><br>
拒绝黑箱 API，拒绝调包侠。我们从 <code>torch.matmul</code> 开始，亲手构建 DeepSeek-V3、Llama-3 等顶尖架构，征服 Pretrain、SFT、LoRA、QLoRA、DPO、GRPO、PPO 的每一座高山。
</p>

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
  <img alt="Author" src="https://img.shields.io/badge/Author-会长-007ACC?style=flat-square">
  <img alt="Framework" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch">
  <img alt="Architecture" src="https://img.shields.io/badge/DeepSeek_V3-Inside-blueviolet?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey?style=flat-square">
</div>

</div>

<br>

---

## 📖 项目愿景 (Vision)

在算法日益黑箱化、高级 API 唾手可得的时代，我们是否还记得那些让现代 AI 成为可能的基石？许多教程要么止步于理论，要么过度依赖 `transformers` 等高级库，让学习者在抽象的 API 调用中“知其然，而不知其所以然”。

本项目旨在打破这一困境，打造一个**终极的、体系化的、从零开始**的大语言模型（LLM）学习与实践平台。我们的核心哲学是：

> **“代码即理论，实践出真知”**
> *(Code is Theory, Practice is Truth)*

我们将以一份详尽的技术蓝图为指引，带领你穿越理论的迷雾，**亲手用纯 PyTorch 实现 LLM 生命周期的每一个核心环节**：从数据处理、分词器构建，到模型架构（Attention, RoPE, SwiGLU, MoE, MLA），再到准工业级训练框架（DDP, Mixed Precision），最后完成监督微调（SFT）与强化学习对齐（RLHF-PPO/DPO/GRPO）。

---

## ⚡ 快速开始 (Quick Start)

我们提供了一键式脚本，为你自动生成经典模型的全套配置。

### 1. 环境准备
```bash
# 强烈建议使用 uv 进行包管理
uv pip install -r requirements.txt
```

### 2. 数据流水线 (只需运行一次)
```bash
# 下载数据 -> 训练分词器 -> 预处理 -> 构建二进制文件 -> 下载 Prompt
python data_pipeline/download/download_tinystories.py && \
python data_pipeline/tokenizer/train_tokenizer.py --vocab_size 4096 --data_limit_mb 100 && \
python data_pipeline/processing/encode_stories.py && \
python data_pipeline/processing/build_pretrain_bins.py && \
python data_pipeline/processing/build_sft_bins.py && \
python data_pipeline/processing/build_preference_bins.py && \
python data_pipeline/download/download_prompts.py
```

### 3. 生成经典模型套件
运行以下命令，自动生成 DeepSeek、Llama、Gemma 的全流程配置文件：
```bash
python utils/setup_classic_models.py
```
*现在，请查看 `configs/classic_reproductions/` 目录，那里有为你准备好的**魔法书**。*

### 4. 启动训练 (以 DeepSeek-V3 Nano 为例)
```bash
# 预训练 (Pretrain) - 支持 torch.compile 加速
python pretrain/train.py --config_path configs/classic_reproductions/deepseek/v3_nano/0_pretrain.yaml --compile

# 监督微调 (SFT - QLoRA)
python finetune/peft/qlora/sft_qlora_train.py --config_path configs/classic_reproductions/deepseek/v3_nano/1_sft_qlora.yaml

# 强化学习对齐 (RLHF - GRPO)
python align/train_online.py --config_path configs/classic_reproductions/deepseek/v3_nano/3_rlhf_grpo.yaml
```

---

## 🗺️ 项目技术栈清单 (Roadmap & Checklist)

这是一个动态更新的清单，展示了我们**已征服 `[x]`** 的领土，和**计划攻克 `[ ]`** 的高地。

### **一、数据工程 (Data Engineering)**

-   [X]  **数据收集**: HuggingFace Dataset Streaming
-   [X]  **Tokenizer**: 
    -   [X]  **BPE** (Byte Pair Encoding) 算法手写实现
    -   [X]  HuggingFace Tokenizers 高性能训练集成
-   [X]  **数据预处理**:
    -   [X]  **Packed Sequences** (无填充高效打包)
    -   [X]  `np.memmap` 零拷贝内存映射
-   [ ]  **高级数据清洗**:
    -   [ ]  MinHash/LSH 去重算法
    -   [ ]  PII (个人敏感信息) 自动去除

### **二、模型架构 (Model Architecture)**

-   [X]  **基础架构**: Transformer (Decoder-only)
-   **注意力机制 (Attention)**:
    -   [X]  **MHA** (Multi-Head Attention - GPT/Llama2)
    -   [X]  **GQA** (Grouped-Query Attention - Llama3)
    -   [X]  **MQA** (Multi-Query Attention - Gemma2)
    -   [X]  **MLA** (Multi-head Latent Attention - **DeepSeek-V2/V3**)
    -   [X]  **Linear Attention** (O(N) 复杂度 / RNN 模式)
    -   [ ]  **Sliding Window Attention** (Longformer/Mistral 风格)
    -   [ ]  **Ring Attention** (超长上下文分布式注意力)
-   **前馈网络 (FFN)**:
    -   [X]  **SwiGLU** (Llama 标准)
    -   [X]  **DeepSeekMoE** (细粒度专家 + 共享专家 + Aux-free LB)
-   **位置编码**:
    -   [X]  **RoPE** (Rotary Positional Embedding) 含 Paged 支持
    -   [X]  **ALiBi** (Linear Biases)
    -   [ ]  **YaRN** (长文本外推插值)
-   **未来架构探索**:
    -   [ ]  **SSM** (State Space Models, e.g., Mamba 2)
    -   [ ]  **Multimodal** (Vision Encoder + Projector 实现多模态)

### **三、训练系统 (Training System)**

-   [X]  **并行训练**: **DDP** (Distributed Data Parallel)
-   [X]  **混合精度**: Bfloat16 / Float16 (GradScaler)
-   [X]  **编译器加速**: `torch.compile` (Inductor Backend)
-   [X]  **优化器**: AdamW, **Muon** (Momentum Orthogonalized)
-   [X]  **稳定性**: 动态梯度裁剪、Loss Spike 检测、Windows 自举
-   [ ]  **高级并行**:
    -   [ ]  **FSDP** (Fully Sharded Data Parallel) - 训练更大的模型
    -   [ ]  **Pipeline Parallelism** (PP) - 流水线并行

### **四、微调与参数高效学习 (SFT & PEFT)**

-   [X]  **Full SFT**: 全量参数微调
-   [X]  **LoRA**: Low-Rank Adaptation (支持自动层名探测)
-   [X]  **QLoRA**: 4-bit NF4 量化 + LoRA
-   [ ]  **Long-Context SFT**: 针对长文本的微调策略 (Packing & Masking)

### **五、人类价值观对齐 (Alignment / RLHF)**

-   [X]  **Reward Modeling**: 奖励模型训练 (Pairwise Loss)
-   [X]  **Offline RL (离线对齐)**:
    -   [X]  **DPO** (Direct Preference Optimization)
    -   [X]  **ORPO** (Odds Ratio Preference Optimization)
    -   [ ]  **KTO** (Kahneman-Tversky Optimization)
-   [X]  **Online RL (在线对齐)**:
    -   [X]  **PPO** (Proximal Policy Optimization) - 完整 GAE
    -   [X]  **GRPO** (Group Relative Policy Optimization) - **DeepSeek 核心**
    -   [X]  **GSPO** (Group Sequence Policy Optimization)
    -   [ ]  **Rejection Sampling** (RFT) - 拒绝采样微调

### **六、推理与评估 (Inference & Eval)**

-   **推理引擎**:
    -   [X]  **Chat**: 流式生成 (Streaming)
    -   [X]  **PagedAttention**: 仿 vLLM 分页内存管理 (适配标准架构)
    -   [X]  **KV Cache**: 支持 Standard & Latent (MLA) 缓存
    -   [X]  **OpenAI API**: 兼容 `/v1/chat/completions`
    -   [ ]  **Speculative Decoding**: 投机采样加速
-   **评估体系**:
    -   [X]  **GSM8K** (数学推理)
    -   [X]  **MMLU** (多任务知识)
    -   [X]  **Perplexity** (困惑度)
    -   [ ]  **HumanEval** (代码能力评估)

---

## ❤️ 欢迎贡献

本项目是一个开放的、持续生长的学习资源。我们深知其中必有不足与疏漏，**我们热切地欢迎任何形式的贡献**！

无论是一个错字的修正、一行注释的补充、一个Bug的修复，还是一个新功能的PR，都是对开源社区的宝贵贡献。请不要犹豫，Fork本项目并发起你的Pull Request吧！

---

<div align="center">
  <br>
  <samp style="font-size: 18px; font-style: italic; color: #555;">
    "追风赶月莫停留，平芜尽处是春山。"
  </samp>
  <br>
  <samp style="font-size: 14px; color: #888;">
    Chasing the wind and the moon, we shall not stay;<br>where the plains end, the verdant mountains of spring await.
  </samp>
</div>
