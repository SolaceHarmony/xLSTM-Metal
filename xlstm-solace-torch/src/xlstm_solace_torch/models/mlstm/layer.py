from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import torch
from torch import nn
import torch.nn.functional as F
from xlstm_torch.kernels.torch.backend_module import (
    mLSTMBackendConfig,
    mLSTMBackend,
)
from ..components import MultiHeadLayerNorm, soft_cap
from xlstm_solace_torch.kernels.torch.recurrent.native_sequence import (
    mlstm_recurrent_sequence__native_fw,
)

WeightModeType = Literal["single", "fused"]

@dataclass
class mLSTMLayerConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""

    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""

    mlstm_backend: mLSTMBackendConfig = field(default_factory=mLSTMBackendConfig)
    """Configuration of the mLSTM backend."""

    weight_mode: WeightModeType = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and output gates.
    """


class mLSTMLayer(nn.Module):
    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        if self.config.weight_mode == "single":
            self.q = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.k = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.v = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )

            self.ogate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )
            self.igate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
            self.fgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.qk_dim + 2 * self.v_dim,
                bias=self.config.use_bias,
            )
            self.ifgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.config.num_heads,
                bias=True,
            )

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)

        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.v_dim // self.config.num_heads,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.v_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape
        if self.config.weight_mode == "single":
            # TorchScript-friendly inter-op parallelism for independent linears
            if torch.jit.is_scripting():
                f_q = torch.jit._fork(F.linear, x, self.q.weight, self.q.bias)
                f_k = torch.jit._fork(F.linear, x, self.k.weight, self.k.bias)
                f_v = torch.jit._fork(F.linear, x, self.v.weight, self.v.bias)
                f_o = torch.jit._fork(F.linear, x, self.ogate_preact.weight, self.ogate_preact.bias)
                f_i = torch.jit._fork(F.linear, x, self.igate_preact.weight, self.igate_preact.bias)
                f_f = torch.jit._fork(F.linear, x, self.fgate_preact.weight, self.fgate_preact.bias)

                q = torch.jit._wait(f_q)
                k = torch.jit._wait(f_k)
                v = torch.jit._wait(f_v)
                o_preact = torch.jit._wait(f_o)
                i_preact = soft_cap(torch.jit._wait(f_i), cap_value=self.config.gate_soft_cap)
                f_preact = soft_cap(torch.jit._wait(f_f), cap_value=self.config.gate_soft_cap)
            else:
                q = self.q(x)
                k = self.k(x)
                v = self.v(x)
                o_preact = self.ogate_preact(x)
                i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
                f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)

        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(
                self.ifgate_preact(x), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state

        # TorchScript path: use native recurrent sequence function to avoid dynamic backend dispatch
        if torch.jit.is_scripting():
            # reshape to (B, NH, S, DH)
            q_seq = q.transpose(1, 2)
            k_seq = k.transpose(1, 2)
            v_seq = v.transpose(1, 2)
            i_seq = i_preact  # already (B, NH, S)
            f_seq = f_preact  # already (B, NH, S)
            if state is None:
                h, state = mlstm_recurrent_sequence__native_fw(
                    q=q_seq,
                    k=k_seq,
                    v=v_seq,
                    i=i_seq,
                    f=f_seq,
                    c_initial=None,
                    n_initial=None,
                    m_initial=None,
                    return_last_states=True,
                    eps=self.config.norm_eps,
                    dtype_state=getattr(torch, self.config.mlstm_backend.inference_state_dtype),
                )
            else:
                c_initial, n_initial, m_initial = state
                h, state = mlstm_recurrent_sequence__native_fw(
                    q=q_seq,
                    k=k_seq,
                    v=v_seq,
                    i=i_seq,
                    f=f_seq,
                    c_initial=c_initial,
                    n_initial=n_initial,
                    m_initial=m_initial,
                    return_last_states=True,
                    eps=self.config.norm_eps,
                    dtype_state=getattr(torch, self.config.mlstm_backend.inference_state_dtype),
                )
        else:
            h, state = self.mlstm_backend(
                q=q,
                k=k,
                v=v,
                i=i_preact,
                f=f_preact,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
            )
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )
        assert (
            h.shape == expected_h_shape
        ), f"Got {h.shape}, expected {expected_h_shape}"

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state
