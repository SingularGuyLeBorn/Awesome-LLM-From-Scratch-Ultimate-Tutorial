# FILE: models/config.py
"""
[v3.2 - DeepSeek-V3 Support]
- 新增 DeepSeek-V3 核心特性参数:
  1. aux_free_lb: 无辅助损失负载均衡
  2. num_shared_experts: 共享专家 (DeepSeekMoE)
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ModelArgs:
    # --- 基础维度 ---
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = -1

    # --- FFN & MoE (DeepSeek-V3 Update) ---
    ffn_hidden_dim: int = None
    multiple_of: int = 256
    norm_eps: float = 1e-5

    num_experts: int = 0
    num_experts_per_tok: int = 2
    moe_layers_indices: Optional[List[int]] = None

    # [DeepSeek-V3 新增]
    num_shared_experts: int = 0  # 共享专家数量 (总是激活)
    use_aux_free_lb: bool = False  # 是否使用无辅助损失负载均衡 (Bias update)

    # --- Attention 变体 ---
    attention_variant: str = "mha"

    # [MLA 参数]
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    nope_head_dim: int = 64
    rope_head_dim: int = 64
    v_head_dim: int = 128

    # [MoBA 参数]
    moba_block_size: int = 512
    moba_topk: int = 2

    # [NSA 参数]
    nsa_compression_block_size: int = 64
    nsa_selection_block_size: int = 128
    nsa_selected_blocks: int = 4
    nsa_sliding_window_size: int = 256

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
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            self.ffn_hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        if self.attention_variant == "mla":
            if self.kv_lora_rank == 0:
                self.kv_lora_rank = 512
            if self.q_lora_rank == 0:
                self.q_lora_rank = 1536

# END OF FILE: models/config.py