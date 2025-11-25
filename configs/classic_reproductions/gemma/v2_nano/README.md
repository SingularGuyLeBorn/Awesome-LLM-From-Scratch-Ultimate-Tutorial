# 📘 炼丹手记: Gemma-2 Nano (The Efficient Speeder)

> **"基于 Google Gemma-2 架构的微缩版。采用 MQA (多查询注意力，KV头数为1)，极大地减少了推理时的 KV Cache 占用，推理速度极快。"**

欢迎来到 LLM 实战复现套件。本指南将带领你完成从**模型出生**到**对齐人类价值观**的完整生命周期。

---

## 🛠️ 序章：准备工作 (Prerequisites)
俗话说“磨刀不误砍柴工”。在开始训练之前，我们需要准备好数据流水线。
请按顺序执行以下命令。每一步都至关重要。

**1. 下载原始数据**
*从 HuggingFace 下载 TinyStories 数据集。这是我们模型的“课本”。*
```bash
python data_pipeline/download/download_tinystories.py
```

**2. 训练分词器 (Tokenizer)**
*训练一个专门用于处理这些数据的 BPE 分词器。它决定了模型如何“阅读”文本。*
```bash
python data_pipeline/tokenizer/train_tokenizer.py --vocab_size 4096 --data_limit_mb 100
```

**3. 编码数据 (Encode)**
*将文本转换为数字序列 (Token IDs)。这是模型唯一能理解的语言。*
```bash
python data_pipeline/processing/encode_stories.py
```

**4. 构建预训练数据 (Pretrain Bins)**
*将 Token 序列打包成高效的二进制文件，供预训练使用。*
```bash
python data_pipeline/processing/build_pretrain_bins.py
```

**5. 构建指令微调数据 (SFT Bins)**
*准备问答对数据，用于教模型听从指令。*
```bash
python data_pipeline/processing/build_sft_bins.py
```

**6. 构建偏好数据 (Preference Bins)**
*准备“好回答 vs 坏回答”的对比数据，用于奖励模型 (RM) 和 DPO 训练。*
```bash
python data_pipeline/processing/build_preference_bins.py
```

**7. 下载评估提示词 (Prompts)**
*下载用于在线强化学习 (PPO/GRPO) 的 Prompt 集合。*
```bash
python data_pipeline/download/download_prompts.py
```

---

## 🧠 第一章：大脑的诞生 (Pre-training)
**目标**：在一个随机初始化的网络中涌现出语言能力。
**核心**：模型阅读大量文本，预测下一个词 (Next Token Prediction)。

**启动命令**:
```bash
python pretrain/train.py --config_path configs/classic_reproductions/gemma/v2_nano/0_pretrain.yaml --fast_dev_run --compile
```
*   `--fast_dev_run`: 快速跑通流程，只训练少量步数。**正式训练时请去掉此参数**。
*   `--compile`: 尝试使用 `torch.compile` 加速。

---

## 🎓 第二章：学会听话 (Supervised Fine-Tuning)
**目标**：将预训练模型的“续写能力”转变为“指令遵循能力”。

### 👑 选项 A: 全量微调 (Full SFT)
*更新所有参数，效果最好。*
```bash
python finetune/full/sft_train.py --config_path configs/classic_reproductions/gemma/v2_nano/1_sft_full.yaml --fast_dev_run
```

### 🗡️ 选项 B: LoRA 微调
*只更新 1% 的参数 (Adapter)。*
```bash
python finetune/peft/lora/sft_lora_train.py --config_path configs/classic_reproductions/gemma/v2_nano/1_sft_lora.yaml --fast_dev_run
```

### 🤏 选项 C: QLoRA (4-bit)
*将大脑压缩到 4-bit 再微调。极度节省内存。*
```bash
python finetune/peft/qlora/sft_qlora_train.py --config_path configs/classic_reproductions/gemma/v2_nano/1_sft_qlora.yaml --fast_dev_run
```

---

## ⚖️ 第三章：注入灵魂 (RLHF & Alignment)
**目标**：让模型的回答更符合人类偏好。

### 3.1 培养裁判 (Reward Model)
*教会一个模型去判断回答的好坏。*
```bash
python align/rm_train.py --config_path configs/classic_reproductions/gemma/v2_nano/2_rm.yaml --fast_dev_run
```

### 3.2 强化学习 (Alignment)
*任选一种流派:*

*   **DPO (离线)**: *直接在偏好数据上优化，无需 RM。*
    ```bash
    python align/train_offline.py --config_path configs/classic_reproductions/gemma/v2_nano/3_rlhf_dpo.yaml --fast_dev_run
    ```

*   **GRPO (在线)**: *DeepSeek 核心科技。无 Critic，自博弈进化。*
    ```bash
    python align/train_online.py --config_path configs/classic_reproductions/gemma/v2_nano/3_rlhf_grpo.yaml --fast_dev_run
    ```

*   **PPO (在线)**: *经典 RLHF 算法。*
    ```bash
    python align/train_online.py --config_path configs/classic_reproductions/gemma/v2_nano/3_rlhf_ppo.yaml --fast_dev_run
    ```

---

## 📝 第四章：期末考试 (Evaluation)
是时候看看我们的“孩子”学得怎么样了。我们在 **GSM8K** (数学) 数据集上进行测试。

**1. 考核预训练模型 (Base Model)**
*看看没受过教育的原始大脑能得几分。*
```bash
python evaluation/run_leaderboard.py --config_path configs/classic_reproductions/gemma/v2_nano/0_pretrain.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth --tasks gsm8k --limit 20
```

**2. 考核 Full SFT 模型**
*看看经过指令微调后，它的逻辑能力是否有提升。*
```bash
python evaluation/run_leaderboard.py --config_path configs/classic_reproductions/gemma/v2_nano/1_sft_full.yaml --checkpoint_path runs/sft/full/fast-dev-run/checkpoints/ckpt_best.pth --tasks gsm8k --limit 20
```

**3. 考核 RLHF 模型 (以 GRPO 为例)**
*看看对齐后的模型表现如何。*
```bash
python evaluation/run_leaderboard.py --config_path configs/classic_reproductions/gemma/v2_nano/3_rlhf_grpo.yaml --checkpoint_path runs/rlhf/online/grpo-fast-dev-run/checkpoints/ckpt_best.pth --tasks gsm8k --limit 20
```

---

## 💬 终章：与它对话 (Chat Inference)
恭喜！你已经走完了全程。

**加载 Base 模型:**
```bash
python inference/chat.py --config_path configs/classic_reproductions/gemma/v2_nano/0_pretrain.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth --quantize
```

**加载 QLoRA 模型:**
```bash
python inference/chat.py --config_path configs/classic_reproductions/gemma/v2_nano/1_sft_qlora.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth --adapter_path runs/sft/peft/qlora/fast-dev-run/checkpoints/ckpt_best.pth
```

## 🌐 第五章：云端部署 (Serving)
让你的模型像 ChatGPT 一样提供 API 服务，支持高并发和 PagedAttention 加速。

**1. 启动 API 服务器**:
```bash
python inference/api_server.py --config_path configs/classic_reproductions/gemma/v2_nano/0_pretrain.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth
```

**2. 发送请求测试**:
*(请另开一个终端运行)*
```bash
python -c "import requests; print(requests.post('http://127.0.0.1:8000/v1/chat/completions', json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello!'}]}).json())"
```


---
*Generated by Project Codex Builder*
