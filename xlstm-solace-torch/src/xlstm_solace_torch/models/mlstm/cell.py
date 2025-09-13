"""
Apple Silicon mLSTM Cell implementation.

This module provides mLSTMCell with Metal acceleration instead of Triton.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass

from ...kernels.torch.registry import get_mlstm_kernel, get_mlstm_step_kernel
from ..components import MultiHeadLayerNorm


@dataclass
class mLSTMCellConfig:
    """Configuration for mLSTM Cell."""
    embedding_dim: int = -1
    num_heads: int = 4
    context_length: int = -1
    
    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.context_length <= 0:
            self.context_length = 2048  # Default context length


def bias_linspace_init_(bias: torch.Tensor, start: float, end: float) -> None:
    """Initialize bias with linearly spaced values."""
    with torch.no_grad():
        bias.copy_(torch.linspace(start, end, bias.shape[0]))


class mLSTMCell(nn.Module):
    """Apple Metal-accelerated mLSTM Cell."""
    
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config

        # Use Apple Metal kernels directly instead of compatibility layer
        self.backend_fn = get_mlstm_kernel("parallel--metal_autograd")
        self.backend_fn_step = get_mlstm_step_kernel("recurrent--metal_autograd")

        self.igate = nn.Linear(3 * config.embedding_dim, config.num_heads)
        self.fgate = nn.Linear(3 * config.embedding_dim, config.num_heads)

        self.outnorm = MultiHeadLayerNorm(ndim=config.embedding_dim, weight=True, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass using Apple Metal acceleration."""
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)

        h_state = self.backend_fn(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=self.causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mlstm_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Single step forward for generation."""
        B, S, _ = q.shape  # (B, S, H)
        assert S == 1, f"mLSTMCell.step only supports sequence length S=1, but got S={S}."

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        _, _, NH, DH = q.shape

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)

        if mlstm_state is None:
            c_state = torch.zeros(size=(B, NH, DH, DH), device=q.device, dtype=q.dtype)
            n_state = torch.zeros(size=(B, NH, DH, 1), device=q.device, dtype=q.dtype)
            m_state = torch.zeros(size=(B, NH, 1, 1), device=q.device, dtype=q.dtype)
        else:
            c_state, n_state, m_state = mlstm_state
            c_state = c_state.to(device=q.device, dtype=q.dtype)
            n_state = n_state.to(device=q.device, dtype=q.dtype)
            m_state = m_state.to(device=q.device, dtype=q.dtype)

        assert c_state.shape == (B, NH, DH, DH), f"Expected c_state shape {(B, NH, DH, DH)}, but got {c_state.shape}."
        assert n_state.shape == (B, NH, DH, 1), f"Expected n_state shape {(B, NH, DH, 1)}, but got {n_state.shape}."
        assert m_state.shape == (B, NH, 1, 1), f"Expected m_state shape {(B, NH, 1, 1)}, but got {m_state.shape}."

        h_state, mlstm_state = self.backend_fn_step(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )  # (B, NH, 1 DH), ((B, NH, DH, DH), (B, NH, DH, 1), (B, NH, 1, 1))

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm, mlstm_state

    def reset_parameters(self):
        """Reset parameters with proper initialization."""
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
