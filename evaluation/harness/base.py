# FILE: evaluation/harness/base.py
"""
评测框架基类。定义所有 Benchmark 必须实现的接口。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class Benchmark(ABC):
    def __init__(self, dataset_name: str, shot_num: int = 0):
        self.dataset_name = dataset_name
        self.shot_num = shot_num

    @abstractmethod
    def load_data(self):
        """加载数据集"""
        pass

    @abstractmethod
    def make_prompt(self, data_sample: Dict) -> str:
        """构建输入的 Prompt (包含 Few-shot 示例)"""
        pass

    @abstractmethod
    def evaluate(self, model, tokenizer, limit: int = None) -> Dict[str, float]:
        """
        执行评测循环。
        Args:
            model: 模型实例
            tokenizer: 分词器
            limit: 限制评测样本数 (用于快速测试)
        Returns:
            Metrics 字典 (e.g. {"accuracy": 0.85})
        """
        pass
# END OF FILE: evaluation/harness/base.py