# FILE: data_pipeline/tokenizer/bpe.py
"""
从零手写BPE (Byte Pair Encoding) 分词器。

此文件包含两个实现:
1.  SimpleTokenizer: 一个简单、清晰的纯Python实现，用于教学和理解算法核心逻辑。
    它的性能较差，不适合处理大规模数据。
2.  RegexTokenizer: 一个性能优化版，其 `encode` 方法利用正则表达式实现闪电般的速度。
    这是我们在项目中实际使用的版本。
"""
import regex as re
from typing import Dict, List, Tuple
from tqdm import tqdm

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """计算相邻对的频率。"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int):
    """用新ID替换字节对。"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# --- 教学版：清晰但慢 ---
class SimpleTokenizer:
    """一个简单、清晰的BPE实现，用于教学目的。"""
    def __init__(self):
        self.merges = {}
        self.vocab = self._build_base_vocab()

    def _build_base_vocab(self): return {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        # ... (训练逻辑与之前的单线程版本完全相同) ...
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        pbar = tqdm(range(num_merges), disable=not verbose, desc="SimpleTokenizer Training")
        for i in pbar:
            stats = get_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> List[int]:
        # ... (编码逻辑也是纯Python循环) ...
        tokens = list(text.encode("utf-8"))
        while True:
            stats = get_stats(tokens)
            if not stats: break # 如果只有一个token，直接退出
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges: break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

# --- 高性能版：训练慢，编码快 ---
class RegexTokenizer:
    """使用正则表达式优化 `encode` 过程的高性能BPE实现。"""
    def __init__(self, pattern: str = None):
        # GPT-4使用的正则表达式模式，能很好地处理各种文本情况
        self.pattern = re.compile(pattern if pattern else r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        self.merges = {}
        self.vocab = self._build_base_vocab()
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _build_base_vocab(self) -> Dict[int, bytes]:
        return {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """优化版训练：一次性处理所有文本，避免重复切分"""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        if verbose: print("1/2: 预处理文本并转换为字节...")
        # 直接把整个文本当作一个大块处理（简化版）
        ids = list(text.encode("utf-8"))

        if verbose: print(f"2/2: 开始 {num_merges} 次合并 (文本长度: {len(ids):,} bytes)...")
        pbar = tqdm(range(num_merges), disable=not verbose, desc="Training")

        for i in pbar:
            # 只计算一次统计
            stats = get_stats(ids)
            if not stats:
                if verbose: print(f"\n提前结束于第 {i} 次合并（无更多可合并对）")
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + i

            # 只执行一次merge
            ids = merge(ids, pair, new_id)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and i % 50 == 0:  # 每50次更新一次显示
                token_preview = self.vocab[new_id].decode('utf-8', errors='replace')[:20]
                pbar.set_postfix_str(f"'{token_preview}' ({stats[pair]}x)", refresh=False)

        if verbose: print("✅ 训练完成")

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """对单个文本块进行编码"""
        ids = list(text_bytes)
        while len(ids) > 1:
            stats = get_stats(ids)
            # 找到在合并规则中排名最靠前（值最小）的对
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            # 如果这个最靠前的对都不在我们的合并规则里，说明不能再合并了
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text: str) -> List[int]:
        """使用正则表达式进行预分词，然后对每个块编码。"""
        tokens = []
        for chunk in re.findall(self.pattern, text):
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self._encode_chunk(chunk_bytes)
            tokens.extend(chunk_ids)
        return tokens


    def decode(self, ids: List[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_prefix: str):
        """将分词器模型保存到文件。"""
        model_file = f"{file_prefix}.model"
        with open(model_file, 'w', encoding="utf-8") as f:
            # 写入版本号和模式
            f.write("regex_bpe_v1\n")
            f.write(f"{self.pattern.pattern}\n")
            # 写入合并规则
            # 按照 token_id 顺序保存
            sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
            for (p1, p2), idx in sorted_merges:
                f.write(f"{p1} {p2}\n")
        if self.special_tokens:
            special_tokens_file = f"{file_prefix}.json"
            import json
            with open(special_tokens_file, 'w', encoding='utf-8') as f:
                json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)
        print(f"✅ 合并规则已保存到 {model_file}")

    def load(self, model_file: str):
        """从文件加载分词器模型。"""
        assert model_file.endswith(".model")
        self.merges = {}
        self.vocab = self._build_base_vocab()
        with open(model_file, 'r', encoding="utf-8") as f:
            # 读取版本号
            version = f.readline().strip()
            assert version == "regex_bpe_v1"
            # 读取正则表达式模式
            self.pattern = re.compile(f.readline().strip())
            # 读取合并规则
            for i, line in enumerate(f):
                p1, p2 = map(int, line.split())
                idx = 256 + i
                self.merges[(p1, p2)] = idx
                self.vocab[idx] = self.vocab[p1] + self.vocab[p2]
        print(f"✅ 从 {model_file} 加载了 {len(self.merges)} 条合并规则。")

# END OF FILE: data_pipeline/tokenizer/bpe.py