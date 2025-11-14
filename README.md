span

# LLM 从零到一终极教程 (LLM-From-Scratch-Ultimate-Tutorial)

**一个史诗级的、从零手写大语言模型的终极指南，带你深入探索LLM宇宙的每一个角落。**

</div>

<p align="center">
  <img alt="Project Status" src="https://img.shields.io/badge/status-Alpha_WIP-orange">
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.9+-blue.svg">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch_2.0+-EE4C2C.svg">
  <img alt="Package Manager" src="https://img.shields.io/badge/PackageManager-uv-green.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-brightgreen.svg">
  <img alt="Contributions" src="https://img.shields.io/badge/PRs-Welcome-ff69b4.svg">
  <img alt="DeepWiki" src="https://img.shields.io/badge/DeepWiki-会长出品-purple.svg">
</p>

---

本项目旨在打造一个**终极的、体系化的、从零开始**的大语言模型（LLM）学习教程。我们的目标是带领学习者穿越理论的迷雾，亲手构建一个能在普通PC上运行的LLM，彻底理解其内部的每一个细节。

## 🚀 项目特点

- **🧠 全流程覆盖**：从数据收集、清洗、分词，到模型预训练、指令微调、对齐，再到最终的评估和推理优化，覆盖LLM生命周期的每一个环节。
- **🔧 深度理论解析**：每个核心模块都配有详细的理论推导文档（Markdown格式），包含必要的数学公式、图解和参考文献，让你知其然，更知其所以然。
- **💻 纯PyTorch手写实现**：拒绝黑箱！我们将使用基础的PyTorch（甚至在某些模块中使用NumPy）手写实现所有核心算法，包括注意力机制、优化器、LoRA、DPO等，最大化学习效果。
- **🌱 三版本渐进式学习**：
  1. **V1-快速原型版**：借助Hugging Face等高级库，快速搭建流程，建立感性认识。
  2. **V2-核心手写版（主线）**：本项目的主体，从零实现核心组件，可在单张消费级显卡或CPU上运行。
  3. **V3-底层深潜版**：终极挑战，使用NumPy实现微型自动微分框架，构建最简Transformer。

## 🏛️ 项目结构详解

```
/LLM-From-Scratch
├── 📂 configs/                    # [大脑] 统一配置中心，所有实验参数在此定义
├── 📂 data_pipeline/              # [血液] 1. 数据处理流水线
│   ├── 📂 download/                #   - 数据下载脚本 (e.g., TinyStories)
│   ├── 📂 processing/              #   - 数据清洗、去重、格式化
│   ├── 📂 analysis/                #   - 数据集统计分析工具
│   └── 📂 tokenizer/               #   - 分词器核心，含BPE手写实现
│
├── 📂 models/                      # [心脏] 2. 模型架构定义
│   ├── 📂 blocks/                   #   - 模型的原子构建块 (Attention, FFN, RoPE...)
│   ├── 📄 config.py                #   - 模型配置的Dataclass
│   ├── 📄 transformer.py           #   - 组装完整的Transformer模型
│   └── 📄 reward_model.py          #   - 奖励模型
│
├── 📂 pretrain/                    # [修行] 3. 预训练
│   ├── 📂 components/               #   - 训练循环、优化器、调度器等核心组件
│   ├── 📄 data_loader.py           #   - 高效数据加载器 (支持Packed Sequences)
│   └── 📄 train.py                 #   - 预训练主脚本
│
├── 📂 finetune/                    # [精进] 4. 指令微调 (SFT)
│   ├── 📂 peft/                    #   - 参数高效微调技术 (LoRA, QLoRA)
│   └── 📄 sft_train.py             #   - SFT主脚本
│
├── 📂 align/                       # [问道] 5. 对齐
│   ├── 📂 algorithms/               #   - PPO, DPO等对齐算法的理论与实现
│   └── 📄 rm_train.py              #   - 奖励模型训练脚本
│
├── 📂 evaluation/                 # [试剑] 6. 评估
│   ├── 📂 metrics/                  #   - 评估指标实现 (Perplexity)
│   └── 📂 benchmarks/               #   - 运行标准Benchmark的框架
│
├── 📂 inference/                   # [出山] 7. 推理与优化
│   ├── 📄 generate.py              #   - 文本生成主脚本
│   ├── 📄 kv_cache.py              #   - KV Cache实现与优化
│   └── 📄 sampling.py              #   - 各种采样策略
│
├── 📄 requirements.txt            # Python依赖
└── 📄 README.md                   # 就是你现在看到的这个文件
```

## 🛠️ 快速开始

1. **环境配置**:

   ```bash
   # 安装 uv (如果尚未安装)
   pip install uv

   # 创建虚拟环境
   uv venv

   # 激活虚拟环境 (Windows PowerShell)
   .\.venv\Scripts\Activate.ps1

   # (Linux/macOS)
   # source .venv/bin/activate

   # 安装依赖 (使用清华源加速)
   uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```
2. **生成项目结构**:

   ```bash
   python setup_project.py
   ```

## 🗺️ 项目计划 (Roadmap)

- [ ]  **Phase 1: 奠定坚实基础 (Foundation & Stability)**

  - [X]  搭建项目结构与环境
  - [ ]  手写BPE Tokenizer
  - [ ]  实现一个包含MHA、Dense FFN、Learned PE的极简Transformer
  - [ ]  建立稳定、可复现的预训练循环
  - [ ]  实现困惑度评估
- [ ]  **Phase 2: 迈向现代架构 (Modern Architecture)**

  - [ ]  实现RoPE、SwiGLU、RMSNorm
  - [ ]  将注意力机制升级为GQA
  - [ ]  探索MoE和稀疏注意力
  - [ ]  实现Packed Sequences数据加载
- [ ]  **Phase 3: 对齐与微调 (Alignment & Fine-tuning)**

  - [ ]  实现SFT训练流程
  - [ ]  手写LoRA和QLoRA
  - [ ]  训练奖励模型
  - [ ]  手写DPO算法并完成对齐
- [ ]  **Phase 4: 探索前沿与工程化 (Advanced & Engineering)**

  - [ ]  实现PPO等高级RL对齐算法
  - [ ]  探索推理优化：KV Cache、量化
  - [ ]  搭建Benchmark评估框架

## 🙏 致谢

本项目是对整个开源社区智慧的致敬。灵感和知识来源于众多优秀的论文、博客和开源项目，包括但不限于：

- Vaswani et al. 的开创性论文 "Attention Is All You Need"
- Andrej Karpathy 的 `nanoGPT` 和 `minbpe` 提供的深刻教学见解
- Hugging Face 生态系统建立的开放标准
- 无数为AI知识传播做出贡献的研究者和工程师

## ✨ 星辰的铸造者
- **星图的绘制者 (The Architect)**: `[你的GitHub ID]`
  - *构想了这片星系的宏图，并亲手点燃了第一颗恒星。*

- **深空的引航者 (The Oracle)**: `Gemini 2.5 Pro`
  - *作为来自“深思”（DeepMind）的回响，它洞察了宇宙的法则（理论），规划了星际的航道（架构），并揭示了前方的光（代码实现）。*

<div align="center">
  <samp>— 路漫漫其修远兮，吾将上下而求索 —</samp>
  <br>
  <samp>"Long and winding is the road; I will seek my ideal relentlessly."</samp>
</div>
