# FILE: models/config.py
"""
[v2.0 - 健壮性重构]
- ModelArgs 现在可以安全地忽略未知的关键字参数。
- 保证了 __post_init__ 逻辑的稳定执行。
"""
from dataclasses import dataclass, field
from typing import Any


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

    # Dropout (通常只在训练时使用)
    dropout: float = 0.0

    # 最大序列长度
    max_seq_len: int = 2048

    # [核心修复] 允许 dataclass 接受并忽略未在字段中定义的额外参数
    def __init__(self, **kwargs):
        # 获取所有已定义的字段名
        defined_fields = {f.name for f in self.__dataclass_fields__.values()}

        # 为所有已定义的字段设置属性
        for field_name in defined_fields:
            # 如果kwargs中提供了值，则使用它，否则使用默认值
            value = kwargs.get(field_name, getattr(self, field_name))
            setattr(self, field_name, value)

        # 调用 __post_init__ （如果存在）
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def __post_init__(self):
        """在对象初始化后自动调用的方法。"""
        # 如果 vocab_size 未设置, 抛出错误
        if self.vocab_size == -1:
            raise ValueError("vocab_size must be set.")

        # 如果 ffn_hidden_dim 未指定，则根据 SwiGLU 规则计算
        if self.ffn_hidden_dim is None:
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            self.ffn_hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

# END OF FILE: models/config.py