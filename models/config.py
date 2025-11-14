# FILE: models/config.py
"""
定义模型架构的配置。
【重构版】将ffn_hidden_dim的计算逻辑封装在内部。
"""
from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    # 维度相关
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = -1  # -1 表示必须在运行时被设置

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

    def __post_init__(self):
        """在对象初始化后自动调用的方法。"""
        # 如果 vocab_size 未设置, 抛出错误
        if self.vocab_size == -1:
            raise ValueError("vocab_size must be set.")

        # 如果 ffn_hidden_dim 未指定，则根据 SwiGLU 规则计算
        if self.ffn_hidden_dim is None:
            # LLaMA 论文中的计算方式
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            # 向上取整到 multiple_of 的最接近的倍数
            self.ffn_hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)


# END OF FILE: models/config.py