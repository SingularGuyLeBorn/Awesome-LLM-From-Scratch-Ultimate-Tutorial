# FILE: inference/engine/paged_engine.py
# -*- coding: utf-8 -*-
"""
[v2.2 - 量化支持] PagedAttention 推理引擎。
- 新增 `quantize` 参数。
"""
import torch
from typing import List, Dict, Tuple
import logging

from tokenizers import Tokenizer
from models.transformer import Transformer
from inference.strategies.sampling import sample
from inference.quantization import Quantizer
from .scheduler import Scheduler, Sequence
from .block_manager import BlockManager


class PagedInferenceEngine:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, block_size: int, num_blocks: int,
                 quantize: bool = False):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        if quantize:
            logging.info("PagedInferenceEngine: 正在应用动态量化...")
            if self.device.type != 'cpu':
                logging.warning("PagedInferenceEngine: 动态量化建议在 CPU 上使用。")
                self.model.to('cpu')
                self.device = torch.device('cpu')

            self.model = Quantizer.quantize_dynamic(self.model)
            logging.info("PagedInferenceEngine: 模型已量化。")

        self.block_size = block_size
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            n_layers=model.args.n_layers,
            n_kv_heads=model.args.n_kv_heads,
            head_dim=model.args.dim // model.args.n_heads,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler = Scheduler(self.block_manager)
        self.eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.max_seq_len = self.model.args.max_seq_len

    def add_request(self, prompt: str, seq_id: int):
        prompt_tokens = self.tokenizer.encode(prompt).ids
        self.scheduler.add_sequence(Sequence(seq_id, prompt_tokens, self.block_size))

    @torch.no_grad()
    def step(self) -> Dict[int, List[int]]:
        scheduled_sequences = self.scheduler.schedule()

        if not scheduled_sequences:
            return {}

        input_tokens, positions, tokens_per_seq, context_lengths, block_tables = self._prepare_model_input(
            scheduled_sequences)

        logits = self.model(
            tokens=input_tokens,
            paged_attention_inputs=(
                positions,
                tokens_per_seq,
                context_lengths,
                self.block_manager.k_cache_pool,
                self.block_manager.v_cache_pool,
                block_tables
            )
        )

        last_token_indices = torch.cumsum(tokens_per_seq, dim=0) - 1
        last_logits = logits[last_token_indices, :]
        next_tokens = sample(last_logits).squeeze(-1)

        finished_outputs = self._process_outputs(scheduled_sequences, next_tokens)
        return finished_outputs

    def _prepare_model_input(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, ...]:
        input_tokens = []
        positions = []
        tokens_per_seq = []
        context_lengths = []

        for seq in sequences:
            unprocessed_tokens = seq.get_unprocessed_tokens()
            num_unprocessed = len(unprocessed_tokens)

            tokens_per_seq.append(num_unprocessed)
            context_lengths.append(seq.get_logical_len())

            input_tokens.extend(unprocessed_tokens)
            positions.extend(range(seq.processed_tokens, seq.get_logical_len()))

            seq.mark_processed(num_unprocessed)

        block_tables = self.scheduler.get_block_tables_tensor(sequences)

        return (
            torch.tensor(input_tokens, device=self.device, dtype=torch.long),
            torch.tensor(positions, device=self.device, dtype=torch.long),
            torch.tensor(tokens_per_seq, device=self.device, dtype=torch.long),
            torch.tensor(context_lengths, device=self.device, dtype=torch.long),
            block_tables
        )

    def _process_outputs(self, sequences: List[Sequence], next_tokens: torch.Tensor) -> Dict[int, List[int]]:
        finished_sequences = {}
        for i, seq in enumerate(sequences):
            new_token_id = next_tokens[i].item()
            seq.append_token(new_token_id)

            is_finished = False
            if new_token_id == self.eos_id:
                is_finished = True

            if seq.get_logical_len() >= self.max_seq_len:
                is_finished = True

            if is_finished:
                finished_sequences[seq.seq_id] = seq.tokens
                self.scheduler.finish_sequence(seq.seq_id)

        return finished_sequences

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_sequences()

# END FILE: inference/engine/paged_engine.py