# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass
from typing import Sequence, List, Optional, Tuple

import torch
from torch import nn

from .components.init import small_init_init_
from .utils import WeightDecayOptimGroupMixin
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False


class xLSTMLMModel(WeightDecayOptimGroupMixin, nn.Module):
    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, **kwargs):
        super().__init__()
        self.config = config

        self.xlstm_block_stack = xLSTMBlockStack(config=config)
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.emb_dropout = nn.Dropout(config.dropout) if config.add_embedding_dropout else nn.Identity()

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()

        small_init_init_(self.token_embedding.weight, dim=self.config.embedding_dim)

        if not self.config.tie_weights:
            small_init_init_(self.lm_head.weight, dim=self.config.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x = self.xlstm_block_stack(x)
        logits = self.lm_head(x)
        return logits

    def step(
        self, idx: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    @torch.jit.export
    def forward_with_states(
        self,
        idx: torch.Tensor,
        mlstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        conv_states: List[Optional[torch.Tensor]],
        slstm_states: List[Optional[torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
    ]:
        """TorchScript-friendly forward with explicit typed states per block."""
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, mlstm_states, conv_states, slstm_states = self.xlstm_block_stack.forward_with_states(
            x, mlstm_states, conv_states, slstm_states
        )
        logits = self.lm_head(x)
        return logits, mlstm_states, conv_states, slstm_states

    @torch.jit.export
    def generate_greedy(
        self,
        prefill_tokens: torch.Tensor,
        max_length: int,
    ) -> Tuple[
        torch.Tensor,
        List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
    ]:
        """Greedy decode using typed per-block states (TorchScript-friendly).

        Returns generated tokens and final typed states.
        """
        device = self.token_embedding.weight.device
        tokens = prefill_tokens.to(device)
        nblocks = len(self.xlstm_block_stack.blocks)
        mlstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [None for _ in range(nblocks)]
        conv_states: List[Optional[torch.Tensor]] = [None for _ in range(nblocks)]
        slstm_states: List[Optional[torch.Tensor]] = [None for _ in range(nblocks)]

        logits, mlstm_states, conv_states, slstm_states = self.forward_with_states(
            tokens, mlstm_states, conv_states, slstm_states
        )
        last = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        outs = [last]
        for _ in range(max_length - 1 if max_length > 0 else 0):
            logits, mlstm_states, conv_states, slstm_states = self.forward_with_states(
                last, mlstm_states, conv_states, slstm_states
            )
            last = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            outs.append(last)
        generated = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return generated, mlstm_states, conv_states, slstm_states

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        # remove token embedding and add it to the correct group, accrording to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weight,)
        else:
            no_weight_decay += (self.token_embedding.weight,)

        return weight_decay, no_weight_decay
