# FILE: models/value_model.py
"""
[v1.1 - 修复版] 价值/评论家模型 (Value/Critic Model) 的架构。
- 修复了从 Transformer 获取输入的数据流。
"""
import torch
import torch.nn as nn
from .transformer import Transformer
from .config import ModelArgs


class ValueModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.transformer = Transformer(args)
        self.value_head = nn.Linear(args.dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回每个 token 位置的价值估计。
        """
        # [核心修复] 明确请求隐藏状态
        hidden_states = self.transformer(tokens, return_hidden_states=True)

        values = self.value_head(hidden_states).squeeze(-1)

        return values
# END OF FILE: models/value_model.py