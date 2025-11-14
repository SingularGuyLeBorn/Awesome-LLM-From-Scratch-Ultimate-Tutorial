# FILE: data_pipeline/tokenizer/teach/bpe_teach.py
"""
【教学版】从零手写BPE (Byte Pair Encoding) 分词器。

此文件包含一个纯Python实现的BPE算法，用于教学和理解其核心逻辑。
它的性能较差，不适合处理大规模数据，仅用于演示。
实际项目中请使用 `data_pipeline/tokenizer/train_tokenizer.py`。
"""
import regex as re
from typing import Dict, List, Tuple
from tqdm import tqdm


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """计算列表中所有相邻元素对的频率。"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """在一个列表中，将所有出现的 `pair` 替换为新的 `idx`。"""
    new_ids = []
    i = 0
    while i < len(ids):
        # 检查当前位置是否是目标对的开始
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2  # 跳过两个元素
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class SimpleTokenizer:
    """
    一个简单、清晰的BPE实现，用于教学目的。
    它在一个大的字节流上进行训练和编码。
    """

    def __init__(self):
        # merges字典存储了合并规则, e.g., {(101, 102): 256}
        self.merges = {}
        # vocab字典存储了从ID到bytes的映射, e.g., {256: b'en'}
        self.vocab = self._build_base_vocab()

    def _build_base_vocab(self) -> Dict[int, bytes]:
        # 初始词汇表就是全部的256个字节
        return {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """在给定的文本上训练BPE模型。"""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 1. 将文本编码为UTF-8字节流
        ids = list(text.encode("utf-8"))

        pbar = tqdm(range(num_merges), disable=not verbose, desc="教学版BPE训练")
        for i in pbar:
            # 2. 计算当前字节流中所有相邻对的频率
            stats = get_stats(ids)
            if not stats:
                if verbose: print("没有更多可合并的对，训练提前结束。")
                break

            # 3. 找到频率最高的对
            pair = max(stats, key=stats.get)

            # 4. 创建新的token ID
            new_id = 256 + i

            # 5. 在字节流中执行合并
            ids = merge(ids, pair, new_id)

            # 6. 存储合并规则和新词元
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                # 实时显示新合并的词元
                token_preview = self.vocab[new_id].decode('utf-8', errors='replace')
                pbar.set_postfix_str(f"新词元: '{token_preview}', 频率: {stats[pair]}", refresh=True)

    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID序列。"""
        tokens = list(text.encode("utf-8"))
        while True:
            # 计算当前token序列的相邻对
            stats = get_stats(tokens)
            if not stats:
                break
            # 找到在合并规则中“最优先”（即最早学会）的合并对
            # 我们用merges字典的值（新ID）来判断优先级，ID越小优先级越高
            # 如果一个对不在merges里，给它一个无穷大的优先级
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break  # 没有可以应用的合并规则了

            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids: List[int]) -> str:
        """将token ID序列解码为文本。"""
        # 从词汇表中查找每个ID对应的字节，然后拼接起来
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # 用UTF-8解码回字符串
        return tokens.decode("utf-8", errors="replace")
# END OF FILE: data_pipeline/tokenizer/teach/bpe_teach.py