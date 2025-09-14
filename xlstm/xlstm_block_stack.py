# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List, Tuple

import torch
from torch import nn
import torch.jit

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
from .components.ln import LayerNorm


@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[mLSTMBlockConfig] = None
    slstm_block: Optional[sLSTMBlockConfig] = None

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: Union[list[int], Literal["all"]] = field(default_factory=list)

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    _block_map: str = None

    @property
    def block_map(self) -> list[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        """Creates the block map, that specifies which block is used at which position."""
        block_map = [0] * self.num_blocks

        for slstm_position_idx in self.slstm_at:
            assert slstm_position_idx < self.num_blocks, f"Invalid slstm position {slstm_position_idx}"
            block_map[slstm_position_idx] = 1

        block_map_str = ",".join(map(str, block_map))

        return block_map_str

    def __post_init__(self):
        if self.mlstm_block is None:
            self.slstm_at = "all"
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))

        if self.mlstm_block is not None:
            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length
            
            self.mlstm_block._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()

        self._block_map = self._create_block_map()


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config

        self.blocks = self._create_blocks(config=config)
        # TorchScript-friendly block type map: 0 = mLSTM, 1 = sLSTM
        self.block_types: List[int] = torch.jit.annotate(List[int], [])
        for v in config.block_map:
            self.block_types.append(int(v))
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(ndim=config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

    def _create_blocks(self, config: xLSTMBlockStackConfig):

        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(mLSTMBlock(config=config))
            elif block_type_int == 1:
                config = deepcopy(self.config.slstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(sLSTMBlock(config=config))
            else:
                raise ValueError(f"Invalid block type {block_type_int}")

        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.post_blocks_norm(x)

        return x

    @torch.jit.export
    def forward_with_states(
        self,
        x: torch.Tensor,
        mlstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        conv_states: List[Optional[torch.Tensor]],
        slstm_states: List[Optional[torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
    ]:
        """TorchScript-friendly forward that carries explicit typed states per block.

        Notes:
            - mLSTM blocks use mlstm_states[i] (c,n,m) and conv_states[i]
            - sLSTM blocks use slstm_states[i] and conv_states[i]
        """
        nblocks = len(self.blocks)
        if len(mlstm_states) != nblocks:
            mlstm_states = [None for _ in range(nblocks)]
        if len(conv_states) != nblocks:
            conv_states = [None for _ in range(nblocks)]
        if len(slstm_states) != nblocks:
            slstm_states = [None for _ in range(nblocks)]

        new_mlstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [None for _ in range(nblocks)]
        new_conv_states: List[Optional[torch.Tensor]] = [None for _ in range(nblocks)]
        new_slstm_states: List[Optional[torch.Tensor]] = [None for _ in range(nblocks)]

        for i in range(nblocks):
            block = self.blocks[i]
            btype = int(self.block_types[i])
            if btype == 0:
                # mLSTM block
                x, st = block.step(
                    x,
                    **{
                        "mlstm_state": mlstm_states[i],
                        "conv_state": conv_states[i],
                    },
                )
                if st is not None:
                    # mLSTMLayer returns dict with keys
                    new_mlstm_states[i] = st.get("mlstm_state")  # type: ignore[attr-defined]
                    new_conv_states[i] = st.get("conv_state")  # type: ignore[attr-defined]
            elif btype == 1:
                # sLSTM block
                x, st = block.step(
                    x,
                    **{
                        "slstm_state": slstm_states[i],
                        "conv_state": conv_states[i],
                    },
                )
                if st is not None:
                    new_slstm_states[i] = st.get("slstm_state")  # type: ignore[attr-defined]
                    new_conv_states[i] = st.get("conv_state")  # type: ignore[attr-defined]
            else:
                raise ValueError(f"Invalid block type {btype}")

        x = self.post_blocks_norm(x)
        return x, new_mlstm_states, new_conv_states, new_slstm_states

    def step(
        self, x: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        if state is None:
            state = {}

        for block_idx, block in enumerate(self.blocks):
            x, state[f"block_{block_idx}"] = block.step(x, **state.get(f"block_{block_idx}", {}))

        x = self.post_blocks_norm(x)

        return x, state
