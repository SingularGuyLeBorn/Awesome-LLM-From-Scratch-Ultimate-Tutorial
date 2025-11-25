# FILE: utils/setup_classic_models.py
# -*- coding: utf-8 -*-
"""
【配置工厂 v3.1 - 深度文档版】
功能：自动生成经典模型 (DeepSeek, Llama, Gemma) 的全生命周期配置。
升级：
1. 数据处理流程分步详解。
2. 评估环节覆盖 Pretrain, SFT, RLHF 全阶段。
"""
import os
import yaml
from pathlib import Path

# --- 1. 模型架构定义 (The Blueprint) ---

ARCHITECTURES = {
    "deepseek": {
        "v3_nano": {
            "title": "DeepSeek-V3 Nano (The Sparse Giant)",
            "description": "基于 DeepSeek-V3 架构的微缩版。核心特性包括 MLA (多头潜变量注意力) 和 DeepSeekMoE (细粒度混合专家)。这是目前最高效、最复杂的架构之一。",
            "support_paged_attention": False,  # MLA 暂不支持 PagedAttention
            "model_config": {
                "dim": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 4,
                "vocab_size": 4096,
                "multiple_of": 16,
                "norm_eps": 1.0e-5,
                "max_seq_len": 128,
                "dropout": 0.0,
                # DeepSeek Specifics
                "attention_variant": "mla",
                "q_lora_rank": 16,
                "kv_lora_rank": 16,
                "v_head_dim": 16,
                "rope_head_dim": 8,
                "nope_head_dim": 8,
                "num_experts": 4,
                "num_shared_experts": 1,
                "num_experts_per_tok": 2,
                "use_aux_free_lb": True,
                "use_activation_checkpointing": True
            },
            "target_modules": "auto"  # MLA 结构复杂，强烈建议自动探测
        }
    },
    "llama": {
        "v3_nano": {
            "title": "Llama-3 Nano (The Robust Standard)",
            "description": "基于 Meta Llama-3 架构的微缩版。采用 GQA (分组查询注意力)、RoPE (旋转位置编码) 和 SwiGLU。这是目前兼容性最好、生态最丰富的架构。",
            "support_paged_attention": True,
            "model_config": {
                "dim": 128,
                "n_layers": 4,
                "n_heads": 4,
                "n_kv_heads": 2,  # GQA: 4 Query Heads, 2 KV Heads
                "vocab_size": 4096,
                "multiple_of": 32,
                "norm_eps": 1.0e-5,
                "max_seq_len": 256,
                "dropout": 0.0,
                "attention_variant": "mha",  # 代码中 mha 兼容 GQA
                "rope_base": 10000,
                "num_experts": 0,
                "use_activation_checkpointing": True
            },
            "target_modules": ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]
        }
    },
    "gemma": {
        "v2_nano": {
            "title": "Gemma-2 Nano (The Efficient Speeder)",
            "description": "基于 Google Gemma-2 架构的微缩版。采用 MQA (多查询注意力，KV头数为1)，极大地减少了推理时的 KV Cache 占用，推理速度极快。",
            "support_paged_attention": True,
            "model_config": {
                "dim": 128,
                "n_layers": 4,
                "n_heads": 4,
                "n_kv_heads": 1,  # MQA: All heads share 1 KV head
                "vocab_size": 4096,
                "multiple_of": 32,
                "norm_eps": 1.0e-6,  # Gemma uses smaller eps
                "max_seq_len": 256,
                "dropout": 0.0,
                "attention_variant": "mha",
                "rope_base": 10000,
                "num_experts": 0,
                "use_activation_checkpointing": True
            },
            "target_modules": ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]
        }
    }
}


# --- 2. 配置生成器 (The Config Factory) ---

def get_base_config(run_name, output_dir="./runs/", device="cpu"):
    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "device": device,
        "console": {"verbose": True}
    }


def get_data_config(stage="pretrain"):
    base = {
        "tokenizer_name": "./data_pipeline/processed_data/tinystories_project_vs4096.json"
    }
    if stage == "pretrain":
        base["data_dir"] = "./data_pipeline/processed_data/"
        base["train_data_limit"] = 5000
        base["val_data_limit"] = 200
    elif stage == "sft":
        base["sft_data_path"] = "./data_pipeline/processed_data/sft_data.bin"
    elif stage in ["rm", "dpo", "orpo"]:
        base["data_dir"] = "./data_pipeline/processed_data/"  # Expects preference bins
    elif stage in ["ppo", "grpo", "gspo"]:
        base["prompt_data_path"] = "./data_pipeline/prompts/h4_prompts.txt"
    return base


def get_training_config(stage="pretrain"):
    # 基础训练参数
    cfg = {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_epochs": 1,
        "weight_decay": 0.01,
        "clip_grad_norm": 1.0,
        "loss_spike_threshold": 10.0,
        "use_activation_checkpointing": True
    }

    # 针对不同阶段的超参微调
    if stage == "pretrain":
        cfg["learning_rate"] = 5.0e-4
        cfg["warmup_ratio"] = 0.1
        cfg["min_lr_ratio"] = 0.1
        cfg["max_epochs"] = 2  # 预训练多跑几轮
    elif stage == "sft":
        cfg["learning_rate"] = 2.0e-5  # SFT 需要更小的 LR
    elif stage == "lora":
        cfg["learning_rate"] = 2.0e-4  # LoRA 可以承受较大的 LR
    elif stage == "qlora":
        cfg["learning_rate"] = 1.5e-4
        cfg["clip_grad_norm"] = 0.3  # QLoRA 梯度裁剪更严格
    elif stage == "rm":
        cfg["batch_size"] = 2
        cfg["learning_rate"] = 1.0e-5
    elif stage in ["dpo", "grpo", "ppo"]:
        cfg["batch_size"] = 1  # RL 在 CPU 上通常只能跑小 Batch
        cfg["learning_rate"] = 5.0e-7  # RL 需要极小的 LR
        cfg["max_epochs"] = 1
        cfg["weight_decay"] = 0.0  # RL 通常不加 weight decay

    return cfg


# --- 3. 主逻辑 (The Builder) ---

def generate_configs():
    root_dir = Path(__file__).parent.parent / "configs" / "classic_reproductions"
    root_dir.mkdir(parents=True, exist_ok=True)

    for family, variants in ARCHITECTURES.items():
        for version, details in variants.items():
            # 1. 创建目录结构
            model_dir = root_dir / family / version
            model_dir.mkdir(parents=True, exist_ok=True)

            model_key = f"{family}-{version}"
            title = details["title"]
            desc = details["description"]
            support_api = details["support_paged_attention"]
            model_cfg = details["model_config"]
            target_modules = details["target_modules"]

            print(f"🛠️  Constructing Suite for: {title} ...")

            # --- 生成 YAML 配置文件 ---

            # 0. Pretrain
            pretrain_cfg = get_base_config(f"pretrain-{model_key}-{{timestamp}}")
            pretrain_cfg["data"] = get_data_config("pretrain")
            pretrain_cfg["model"] = model_cfg
            pretrain_cfg["training"] = get_training_config("pretrain")
            pretrain_cfg["logging"] = {"log_interval": 5}
            pretrain_cfg["checkpointing"] = {"save_interval": 200, "resume_from": "none"}
            with open(model_dir / "0_pretrain.yaml", "w", encoding="utf-8") as f:
                yaml.dump(pretrain_cfg, f, sort_keys=False, allow_unicode=True)

            # 1. SFT (Full)
            sft_cfg = get_base_config(f"sft-full-{model_key}-{{timestamp}}")
            sft_cfg["data"] = get_data_config("sft")
            sft_cfg["model"] = model_cfg
            sft_cfg["sft"] = {"base_model_checkpoint": "will_be_overridden_by_fast_dev_run"}
            sft_cfg["training"] = get_training_config("sft")
            sft_cfg["logging"] = {"log_interval": 1}
            sft_cfg["checkpointing"] = {"save_interval": 100}
            with open(model_dir / "1_sft_full.yaml", "w", encoding="utf-8") as f:
                yaml.dump(sft_cfg, f, sort_keys=False, allow_unicode=True)

            # 2. SFT (LoRA)
            lora_cfg = get_base_config(f"sft-lora-{model_key}-{{timestamp}}")
            lora_cfg["data"] = get_data_config("sft")
            lora_cfg["model"] = model_cfg
            lora_cfg["sft"] = {"base_model_checkpoint": "will_be_overridden_by_fast_dev_run"}
            lora_cfg["lora"] = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": target_modules}
            lora_cfg["training"] = get_training_config("lora")
            lora_cfg["logging"] = {"log_interval": 1}
            lora_cfg["checkpointing"] = {"save_interval": 100}
            with open(model_dir / "1_sft_lora.yaml", "w", encoding="utf-8") as f:
                yaml.dump(lora_cfg, f, sort_keys=False, allow_unicode=True)

            # 3. SFT (QLoRA)
            qlora_cfg = get_base_config(f"sft-qlora-{model_key}-{{timestamp}}")
            qlora_cfg["data"] = get_data_config("sft")
            qlora_cfg["model"] = model_cfg
            qlora_cfg["sft"] = {"base_model_checkpoint": "will_be_overridden_by_fast_dev_run"}
            qlora_cfg["qlora"] = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": target_modules,
                                  "compute_dtype": "float32"}
            qlora_cfg["training"] = get_training_config("qlora")
            qlora_cfg["logging"] = {"log_interval": 1}
            qlora_cfg["checkpointing"] = {"save_interval": 100}
            with open(model_dir / "1_sft_qlora.yaml", "w", encoding="utf-8") as f:
                yaml.dump(qlora_cfg, f, sort_keys=False, allow_unicode=True)

            # 4. Reward Model
            rm_cfg = get_base_config(f"rm-{model_key}-{{timestamp}}")
            rm_cfg["data"] = get_data_config("rm")
            rm_cfg["model"] = model_cfg
            rm_cfg["rm"] = {"sft_model_checkpoint": "will_be_overridden_by_fast_dev_run"}
            rm_cfg["training"] = get_training_config("rm")
            rm_cfg["logging"] = {"log_interval": 1}
            with open(model_dir / "2_rm.yaml", "w", encoding="utf-8") as f:
                yaml.dump(rm_cfg, f, sort_keys=False, allow_unicode=True)

            # 5. RLHF (DPO)
            dpo_cfg = get_base_config(f"dpo-{model_key}-{{timestamp}}")
            dpo_cfg["data"] = get_data_config("dpo")
            dpo_cfg["model"] = model_cfg
            dpo_cfg["offline"] = {"algorithm": "dpo", "sft_model_checkpoint": "will_be_overridden", "beta": 0.1,
                                  "label_smoothing": 0.0}
            dpo_cfg["training"] = get_training_config("dpo")
            dpo_cfg["logging"] = {"log_interval": 1}
            with open(model_dir / "3_rlhf_dpo.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dpo_cfg, f, sort_keys=False, allow_unicode=True)

            # 6. RLHF (GRPO)
            grpo_cfg = get_base_config(f"grpo-{model_key}-{{timestamp}}")
            grpo_cfg["data"] = get_data_config("grpo")
            grpo_cfg["model"] = model_cfg
            grpo_cfg["rl"] = {
                "algorithm": "grpo", "sft_model_checkpoint": "will_be_overridden",
                "reward_model_checkpoint": "will_be_overridden",
                "kl_coeff": 0.04, "group_size": 4, "max_prompt_len": 32, "max_gen_len": 64,
                "generate": {"temperature": 0.7, "top_k": 20},
                "rollout_batches": 2, "update_epochs": 1, "update_batch_size": 4, "clip_epsilon": 0.2
            }
            grpo_cfg["training"] = get_training_config("grpo")
            with open(model_dir / "3_rlhf_grpo.yaml", "w", encoding="utf-8") as f:
                yaml.dump(grpo_cfg, f, sort_keys=False, allow_unicode=True)

            # 7. RLHF (PPO)
            ppo_cfg = get_base_config(f"ppo-{model_key}-{{timestamp}}")
            ppo_cfg["data"] = get_data_config("ppo")
            ppo_cfg["model"] = model_cfg
            ppo_cfg["rl"] = {
                "algorithm": "ppo", "sft_model_checkpoint": "will_be_overridden",
                "reward_model_checkpoint": "will_be_overridden",
                "kl_coeff": 0.02, "group_size": 1, "max_prompt_len": 32, "max_gen_len": 64,
                "generate": {"temperature": 0.7, "top_k": 20},
                "rollout_batches": 4, "update_epochs": 1, "update_batch_size": 4,
                "clip_epsilon": 0.2, "gamma": 0.99, "lambda_gae": 0.95, "value_loss_coef": 0.5, "entropy_coef": 0.01
            }
            ppo_cfg["training"] = get_training_config("ppo")
            with open(model_dir / "3_rlhf_ppo.yaml", "w", encoding="utf-8") as f:
                yaml.dump(ppo_cfg, f, sort_keys=False, allow_unicode=True)

            # --- 生成文档 (Story Mode) ---

            api_section = ""
            if support_api:
                api_section = f"""
## 🌐 第五章：云端部署 (Serving)
让你的模型像 ChatGPT 一样提供 API 服务，支持高并发和 PagedAttention 加速。

**1. 启动 API 服务器**:
```bash
python inference/api_server.py --config_path configs/classic_reproductions/{family}/{version}/0_pretrain.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth
```

**2. 发送请求测试**:
*(请另开一个终端运行)*
```bash
python -c "import requests; print(requests.post('http://127.0.0.1:8000/v1/chat/completions', json={{'model': 'test', 'messages': [{{'role': 'user', 'content': 'Hello!'}}]}}).json())"
```
"""
            else:
                api_section = f"""
## 🌐 第五章：云端部署 (Serving)
*(⚠️ 注意：当前模型架构 `{model_cfg['attention_variant'].upper()}` 采用非标准 KV 结构，暂未适配 PagedAttention 引擎。请使用 Chat 模式进行本地交互。)*
"""

            # 完整的 README 内容
            readme_content = f"""# 📘 炼丹手记: {title}

> **"{desc}"**

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
python pretrain/train.py --config_path configs/classic_reproductions/{family}/{version}/0_pretrain.yaml --fast_dev_run --compile
```
*   `--fast_dev_run`: 快速跑通流程，只训练少量步数。**正式训练时请去掉此参数**。
*   `--compile`: 尝试使用 `torch.compile` 加速。

---

## 🎓 第二章：学会听话 (Supervised Fine-Tuning)
**目标**：将预训练模型的“续写能力”转变为“指令遵循能力”。

### 👑 选项 A: 全量微调 (Full SFT)
*更新所有参数，效果最好。*
```bash
python finetune/full/sft_train.py --config_path configs/classic_reproductions/{family}/{version}/1_sft_full.yaml --fast_dev_run
```

### 🗡️ 选项 B: LoRA 微调
*只更新 1% 的参数 (Adapter)。*
```bash
python finetune/peft/lora/sft_lora_train.py --config_path configs/classic_reproductions/{family}/{version}/1_sft_lora.yaml --fast_dev_run
```

### 🤏 选项 C: QLoRA (4-bit)
*将大脑压缩到 4-bit 再微调。极度节省内存。*
```bash
python finetune/peft/qlora/sft_qlora_train.py --config_path configs/classic_reproductions/{family}/{version}/1_sft_qlora.yaml --fast_dev_run
```

---

## ⚖️ 第三章：注入灵魂 (RLHF & Alignment)
**目标**：让模型的回答更符合人类偏好。

### 3.1 培养裁判 (Reward Model)
*教会一个模型去判断回答的好坏。*
```bash
python align/rm_train.py --config_path configs/classic_reproductions/{family}/{version}/2_rm.yaml --fast_dev_run
```

### 3.2 强化学习 (Alignment)
*任选一种流派:*

*   **DPO (离线)**: *直接在偏好数据上优化，无需 RM。*
    ```bash
    python align/train_offline.py --config_path configs/classic_reproductions/{family}/{version}/3_rlhf_dpo.yaml --fast_dev_run
    ```

*   **GRPO (在线)**: *DeepSeek 核心科技。无 Critic，自博弈进化。*
    ```bash
    python align/train_online.py --config_path configs/classic_reproductions/{family}/{version}/3_rlhf_grpo.yaml --fast_dev_run
    ```

*   **PPO (在线)**: *经典 RLHF 算法。*
    ```bash
    python align/train_online.py --config_path configs/classic_reproductions/{family}/{version}/3_rlhf_ppo.yaml --fast_dev_run
    ```

---

## 📝 第四章：期末考试 (Evaluation)
是时候看看我们的“孩子”学得怎么样了。我们在 **GSM8K** (数学) 数据集上进行测试。

**1. 考核预训练模型 (Base Model)**
*看看没受过教育的原始大脑能得几分。*
```bash
python evaluation/run_leaderboard.py --config_path configs/classic_reproductions/{family}/{version}/0_pretrain.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth --tasks gsm8k --limit 20
```

**2. 考核 Full SFT 模型**
*看看经过指令微调后，它的逻辑能力是否有提升。*
```bash
python evaluation/run_leaderboard.py --config_path configs/classic_reproductions/{family}/{version}/1_sft_full.yaml --checkpoint_path runs/sft/full/fast-dev-run/checkpoints/ckpt_best.pth --tasks gsm8k --limit 20
```

**3. 考核 RLHF 模型 (以 GRPO 为例)**
*看看对齐后的模型表现如何。*
```bash
python evaluation/run_leaderboard.py --config_path configs/classic_reproductions/{family}/{version}/3_rlhf_grpo.yaml --checkpoint_path runs/rlhf/online/grpo-fast-dev-run/checkpoints/ckpt_best.pth --tasks gsm8k --limit 20
```

---

## 💬 终章：与它对话 (Chat Inference)
恭喜！你已经走完了全程。

**加载 Base 模型:**
```bash
python inference/chat.py --config_path configs/classic_reproductions/{family}/{version}/0_pretrain.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth --quantize
```

**加载 QLoRA 模型:**
```bash
python inference/chat.py --config_path configs/classic_reproductions/{family}/{version}/1_sft_qlora.yaml --checkpoint_path runs/pretrain/fast-dev-run/checkpoints/ckpt_best.pth --adapter_path runs/sft/peft/qlora/fast-dev-run/checkpoints/ckpt_best.pth
```
{api_section}

---
*Generated by Project Codex Builder*
"""
            # 写入 README.md
            with open(model_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)

    print("\n✅ Classic Reproduction Suites generated successfully!")
    print(f"📂 Explore them at: {root_dir}")


if __name__ == "__main__":
    generate_configs()
# END OF FILE: utils/setup_classic_models.py
