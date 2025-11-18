# FILE: inference/engine/scheduler.py
# -*- coding: utf-8 -*-
"""
[v1.2 - Debugger 版] 推理调度器与序列表示。

- [调试增强] 增加了print语句来追踪序列状态的转换。
- [核心重构] 彻底改变了序列状态的管理方式。移除了不可靠的 `is_prefill()`
  方法，引入了 `processed_tokens` 计数器。现在，一个序列需要处理哪些
  token，是根据其 `processed_tokens` 和 `len(tokens)` 的差值精确计算
  出来的，从根本上解决了 Prefill 和 Decode 阶段的逻辑混淆问题。
"""
import torch
from enum import Enum
from typing import List, Dict
from .block_manager import BlockManager


class SequenceStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3


class Sequence:
    """代表一个独立的推理请求。"""

    def __init__(self, seq_id: int, prompt_tokens: List[int], block_size: int):
        self.seq_id = seq_id
        self.tokens = prompt_tokens
        self.status = SequenceStatus.WAITING
        self.block_size = block_size
        self.block_table: List[int] = []

        # [核心新增] 精确追踪已处理的token数量
        self.processed_tokens = 0

    def append_token(self, token_id: int):
        self.tokens.append(token_id)

    def get_logical_len(self) -> int:
        return len(self.tokens)

    def get_unprocessed_tokens(self) -> List[int]:
        return self.tokens[self.processed_tokens:]

    def mark_processed(self, num_processed: int):
        self.processed_tokens += num_processed

    def get_num_logical_blocks(self) -> int:
        return (self.get_logical_len() + self.block_size - 1) // self.block_size


class Scheduler:
    """管理所有序列的状态，并决定每个推理步骤要运行哪些序列。"""

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.finished: Dict[int, Sequence] = {}

    def add_sequence(self, seq: Sequence):
        print(f"[DEBUG-SCHEDULER] Adding sequence {seq.seq_id} to WAITING queue.")
        self.waiting.append(seq)

    def schedule(self) -> List[Sequence]:
        """
        调度器的核心逻辑。
        """
        print("[DEBUG-SCHEDULER] Running schedule...")
        print(f"  > Before: Waiting={[s.seq_id for s in self.waiting]}, Running={[s.seq_id for s in self.running]}")

        # 将等待队列中的序列按先来后到的顺序排序
        self.waiting.sort(key=lambda s: s.seq_id)

        # 将可以分配资源的等待序列移动到运行队列
        temp_waiting = []
        newly_running_ids = []
        for seq in self.waiting:
            required_blocks = seq.get_num_logical_blocks()
            if self.block_manager.can_allocate(required_blocks):
                seq.status = SequenceStatus.RUNNING
                self.block_manager.allocate(seq.seq_id, required_blocks)
                seq.block_table = self.block_manager.get_block_table(seq.seq_id)
                self.running.append(seq)
                newly_running_ids.append(seq.seq_id)
            else:
                temp_waiting.append(seq)

        if newly_running_ids:
            print(f"  > Moved sequences to RUNNING: {newly_running_ids}")
        self.waiting = temp_waiting

        # 确保正在运行的序列有足够的块
        for seq in self.running:
            required_blocks = seq.get_num_logical_blocks()
            if len(seq.block_table) < required_blocks:
                if self.block_manager.can_allocate(1):
                    self.block_manager.append_block(seq.seq_id)
                    seq.block_table = self.block_manager.get_block_table(seq.seq_id)
                    print(f"  > Appended a block to running sequence {seq.seq_id}.")

        print(f"  > After: Waiting={[s.seq_id for s in self.waiting]}, Running={[s.seq_id for s in self.running]}")
        return self.running

    def finish_sequence(self, seq_id: int):
        seq_to_remove = None
        for seq in self.running:
            if seq.seq_id == seq_id:
                seq_to_remove = seq
                break

        if seq_to_remove:
            self.running.remove(seq_to_remove)
            seq_to_remove.status = SequenceStatus.FINISHED
            self.finished[seq_id] = seq_to_remove
            self.block_manager.free(seq_id)
            print(f"[DEBUG-SCHEDULER] Finishing sequence {seq_id}. Freed its blocks.")

    def has_unfinished_sequences(self) -> bool:
        return bool(self.running or self.waiting)

    def get_block_tables_tensor(self, sequences: List[Sequence]) -> torch.Tensor:
        max_blocks = max(len(seq.block_table) for seq in sequences) if sequences else 0

        tables = []
        for seq in sequences:
            table = seq.block_table
            padded_table = table + [0] * (max_blocks - len(table))
            tables.append(padded_table)# FILE: inference/engine/scheduler.py
# -*- coding: utf-8 -*-
"""
[v1.3 - Cleaned Version] 推理调度器与序列表示。

- 移除了所有调试用的 print 语句。
"""
import torch
from enum import Enum
from typing import List, Dict
from .block_manager import BlockManager


class SequenceStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3


class Sequence:
    """代表一个独立的推理请求。"""

    def __init__(self, seq_id: int, prompt_tokens: List[int], block_size: int):
        self.seq_id = seq_id
        self.tokens = prompt_tokens
        self.status = SequenceStatus.WAITING
        self.block_size = block_size
        self.block_table: List[int] = []
        self.processed_tokens = 0

    def append_token(self, token_id: int):
        self.tokens.append(token_id)

    def get_logical_len(self) -> int:
        return len(self.tokens)

    def get_unprocessed_tokens(self) -> List[int]:
        return self.tokens[self.processed_tokens:]

    def mark_processed(self, num_processed: int):
        self.processed_tokens += num_processed

    def get_num_logical_blocks(self) -> int:
        return (self.get_logical_len() + self.block_size - 1) // self.block_size


class Scheduler:
    """管理所有序列的状态，并决定每个推理步骤要运行哪些序列。"""

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.finished: Dict[int, Sequence] = {}

    def add_sequence(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> List[Sequence]:
        """
        调度器的核心逻辑。
        """
        self.waiting.sort(key=lambda s: s.seq_id)

        temp_waiting = []
        for seq in self.waiting:
            required_blocks = seq.get_num_logical_blocks()
            if self.block_manager.can_allocate(required_blocks):
                seq.status = SequenceStatus.RUNNING
                self.block_manager.allocate(seq.seq_id, required_blocks)
                seq.block_table = self.block_manager.get_block_table(seq.seq_id)
                self.running.append(seq)
            else:
                temp_waiting.append(seq)
        self.waiting = temp_waiting

        for seq in self.running:
            required_blocks = seq.get_num_logical_blocks()
            if len(seq.block_table) < required_blocks:
                if self.block_manager.can_allocate(1):
                    self.block_manager.append_block(seq.seq_id)
                    seq.block_table = self.block_manager.get_block_table(seq.seq_id)

        return self.running

    def finish_sequence(self, seq_id: int):
        seq_to_remove = None
        for seq in self.running:
            if seq.seq_id == seq_id:
                seq_to_remove = seq
                break

        if seq_to_remove:
            self.running.remove(seq_to_remove)
            seq_to_remove.status = SequenceStatus.FINISHED
            self.finished[seq_id] = seq_to_remove
            self.block_manager.free(seq_id)

    def has_unfinished_sequences(self) -> bool:
        return bool(self.running or self.waiting)

    def get_block_tables_tensor(self, sequences: List[Sequence]) -> torch.Tensor:
        max_blocks = max(len(seq.block_table) for seq in sequences) if sequences else 0

        tables = []
        for seq in sequences:
            table = seq.block_table
            padded_table = table + [0] * (max_blocks - len(table))
            tables.append(padded_table)

        return torch.tensor(tables, device=self.block_manager.device, dtype=torch.int)

# END FILE inference/engine/scheduler.py

        return torch.tensor(tables, device=self.block_manager.device, dtype=torch.int)