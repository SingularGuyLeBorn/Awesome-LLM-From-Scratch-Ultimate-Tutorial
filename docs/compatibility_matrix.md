# FILE: docs/compatibility_matrix.md
# 模型架构与功能兼容性矩阵 (Architecture Compatibility Matrix)

本项目集成了从经典的 Transformer 到最前沿的 DeepSeek-V2 和 Native Sparse Attention (NSA) 等多种架构。
由于不同架构的底层计算逻辑（特别是 KV Cache 和 Attention 机制）差异巨大，并非所有后训练和推理工具都能通用。

请在运行 `inference/api_server.py` (PagedAttention) 或 `align/train_online.py` (PPO/GRPO) 前查阅此表。

| 架构类型 (Architecture) | 配置文件示例 (Config Example) | 预训练 (Pretrain) | SFT / LoRA / QLoRA | 标准推理 (Chat.py) | Paged推理 (API Server) | 在线 RLHF (PPO/GRPO) | 备注 (Notes) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard MHA** | `1.4M_pretrain_fast.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | 最稳定，兼容性最好 (Llama2/3 风格) |
| **GQA** | `llama2_gqa_nano.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | 推理速度快，显存占用低 |
| **MQA** | (可通过参数配置) | ✅ | ✅ | ✅ | ✅ | ✅ | 极致 KV Cache 压缩 |
| **DeepSeek-V2 (MLA)** | `deepseek_v2_nano.yaml` | ✅ | ✅ (自动探测层) | ✅ (矩阵吸收优化) | ✅ | ✅ | **推荐**。推理需使用 `LatentKVCache` |
| **MoE (Mixtral)** | `1.4M_moe_fast.yaml` | ✅ | ✅ (自动探测层) | ✅ | ✅ | ✅ | 稀疏激活，训练快，显存大 |
| **Linear Attention** | `linear_attn_nano.yaml` | ✅ | ✅ | ✅ | ❌ | ❌ | 无 Softmax，不支持 PagedAttention |
| **MoBA / NSA** | `nsa_nano.yaml` | ✅ | ✅ | ✅ | ❌ | ❌ | 稀疏注意力，不支持 PagedAttention |

### 图例说明
*   ✅ **支持 (Supported)**: 该功能已针对此架构进行了适配和测试。
*   ❌ **不支持 (Not Supported)**: 该架构的数学原理（如无 Softmax、非标准 KV 结构）与该功能的底层实现（如 vLLM Paged Block 管理）冲突，强行运行会报错或崩溃。
*   **自动探测层**: LoRA/QLoRA 脚本已升级，能自动识别 `w_gate`, `wq_down` 等非标准层名，无需手动修改 YAML。
*   **矩阵吸收优化**: MLA 架构在标准推理中启用了 DeepSeek 论文中的 Matrix Absorption 优化，无需解压 KV 即可推理。