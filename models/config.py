# FILE: models/config.py
"""
[v2.2 - MoE 支持]
- ModelArgs 新增 MoE 相关参数。
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ModelArgs:
    # 维度相关
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = -1

    # FFN 中间层的维度 (如果为None则自动计算)
    ffn_hidden_dim: int = None
    multiple_of: int = 256

    # Normalization 相关
    norm_eps: float = 1e-5

    # RoPE 相关
    rope_base: int = 10000

    # Dropout
    dropout: float = 0.0

    # 最大序列长度
    max_seq_len: int = 2048

    # 训练优化
    use_activation_checkpointing: bool = False

    # [MoE 核心配置]
    num_experts: int = 0  # 专家总数。如果为 0 或 1，则使用标准 FFN
    num_experts_per_tok: int = 2  # 每个 token 选择的专家数 (Top-K)

    # [MoE 混合配置]
    # 允许指定哪些层使用 MoE，哪些层使用 Dense。
    # 这是一个 indices 列表，例如 [2, 4, 6, 8...]
    # 如果为 None 且 num_experts > 1，则默认所有层都使用 MoE
    moe_layers_indices: Optional[List[int]] = None

    # 允许 dataclass 接受并忽略未在字段中定义的额外参数
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

# END OF FILE: models/config.py