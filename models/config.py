# FILE: models/config.py
"""
[v3.0 - Attention Zoo]
- 新增 attention_variant 字段，支持 "mha", "mla", "linear", "moba"。
- 新增 MLA (DeepSeek-V2) 相关参数：q_lora_rank, kv_lora_rank, v_head_dim 等。
- 新增 MoBA 相关参数：block_size, topk_blocks。
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ModelArgs:
    # --- 基础维度 ---
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8  # MHA: n_kv=n_heads; GQA: 1 < n_kv < n_heads; MQA: n_kv=1
    vocab_size: int = -1

    # --- FFN & MoE 配置 ---
    ffn_hidden_dim: int = None
    multiple_of: int = 256
    norm_eps: float = 1e-5

    # MoE
    num_experts: int = 0  # 0=Dense, >1=MoE
    num_experts_per_tok: int = 2
    moe_layers_indices: Optional[List[int]] = None

    # --- Attention 变体配置 ---
    # 选项: "mha" (含GQA/MQA), "mla" (DeepSeek-V2), "linear", "moba"
    attention_variant: str = "mha"

    # [MLA 特有参数] (DeepSeek-V2)
    q_lora_rank: int = 0  # Query 压缩维度 (0表示不压缩)
    kv_lora_rank: int = 0  # KV 压缩维度 (0表示不压缩)
    nope_head_dim: int = 64  # 不参与 RoPE 的头部维度 (content)
    rope_head_dim: int = 64  # 参与 RoPE 的头部维度 (position)
    v_head_dim: int = 128  # Value 的头部维度 (通常比 Q/K 大)

    # [MoBA 特有参数] (Mixture of Block Attention)
    moba_block_size: int = 512
    moba_topk: int = 2

    # --- 位置编码 ---
    rope_base: int = 10000
    max_seq_len: int = 2048

    # --- 训练优化 ---
    dropout: float = 0.0
    use_activation_checkpointing: bool = False

    def __init__(self, **kwargs):
        defined_fields = {f.name for f in self.__dataclass_fields__.values()}
        for field_name in defined_fields:
            value = kwargs.get(field_name, getattr(self, field_name))
            setattr(self, field_name, value)
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def __post_init__(self):
        if self.vocab_size == -1:
            raise ValueError("vocab_size must be set.")

        if self.ffn_hidden_dim is None:
            # SwiGLU 默认建议
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            self.ffn_hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        # MLA 默认参数校验
        if self.attention_variant == "mla":
            if self.kv_lora_rank == 0:
                # 默认设置为 hidden_dim 的一部分，例如 512
                self.kv_lora_rank = 512
            if self.q_lora_rank == 0:
                self.q_lora_rank = 1536

# END OF FILE: models/config.py