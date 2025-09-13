from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterable, List, Tuple

try:
    from xlstm_torch.kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
        ChunkwiseKernelType,
        SequenceKernelType,
        StepKernelType,
        DtypeType,
        BackendModeType,
    )
except ImportError:
    raise ImportError("Kernel backends not found. Ensure xlstm_torch.kernels.torch is included in the package.")


import torch
from torch import nn
from typing import Literal
from .components import MultiHeadLayerNorm, RMSNorm, soft_cap
from .utils import round_up_to_next_multiple_of
from .generate import generate_tokens, get_sampling_fn

mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]
# TorchScript-friendly state: fixed-length list of optional tuples (one per block)
TSStateType = List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]

WeightModeType = Literal["single", "fused"]


@dataclass
class xLSTMTorchConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    num_blocks: int
    """Number of blocks."""
    vocab_size: int
    """Vocabulary size."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""
    add_out_norm: bool = True
    """Whether to add a normalization layer after the block stack."""

    # mlstm layer
    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""

    # mlstm backend
    chunkwise_kernel: ChunkwiseKernelType = "chunkwise--triton_limit_chunk"
    """Kernel to use for chunkwise parallel processing of the sequence.
    Also supports fully parallel (i.e. quadratic) backends for comparison.
    E.g. 'parallel--native_autograd'.
    """
    sequence_kernel: SequenceKernelType = "native_sequence__triton"
    """The sequence kernel to use for processing sequneces step-by-step.
    Used only for parts of the prefill sequence in inference mode.
    """
    step_kernel: StepKernelType = "triton"
    """The step kernel to use for processing a single step.
    Used for generation in inference mode.
    """
    mode: BackendModeType = "train"
    """The mode of operation for the backend. Determines how the `forward` method behaves.
    Available modes are 'train', 'train_with_padding', 'inference'.
    'inference' works with arbitrary sequence lengths, and does not support training. 
    It calls a sequence of different kernels to process the sequence.
    'train_with_padding' pads the input to multiples of `chunk_size`.
    """
    chunk_size: int = 64
    """The chunk size of the chunkwise kernel.
    If `mode` is 'train_with_padding', the inputs are padded to multiples of this size.
    """
    return_last_states: bool = False
    """Whether to return the last states of the sequence in training mode.
    Inference mode always returns the last states.
    """
    autocast_kernel_dtype: DtypeType = "bfloat16"
    """The dtype to use for autocast behavior in the kernel.
    If autocast is enabled all inputs are cast to this dtype before the kernel is called.
    """
    eps: float = 1e-6
    """Epsilon value for numerical stability in the kernel."""
    inference_state_dtype: DtypeType = "float32"
    """The dtype to use for the state tensors in inference mode."""
    # feedforward
    ffn_proj_factor: float = 2.6667
    """The factor to determine the dimension of the intermediate projection in the feedforward layer."""
    ffn_round_up_to_multiple_of: int = 64
    """Round the intermediate projection dimension to the next multiple of this value."""
    
    # capping
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""
    output_logit_soft_cap: float = 30.0
    """Soft cap value for the output logits."""

    weight_mode: WeightModeType = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and gates.
    'fused' is benefitial in inference settings.
    """
    # Runtime/scheduling options for kernels (Solace production)
    runtime_opts: dict = field(default_factory=dict)

    # Optional: control which block indices apply FFN (channel mixer).
    # If None, FFN is applied in every block (current behavior).
    # If a sequence of indices, only those blocks will include FFN; others will skip FFN.
    ffn_blocks: Optional[Iterable[int]] = None


class xLSTMTorch(nn.Module):
    config_class = xLSTMTorchConfig

    def __init__(self, config: xLSTMTorchConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.backbone = xLSTMBlockStack(config)

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim, out_features=config.vocab_size, bias=False
        )

    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, mLSTMStateType]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, S].
            state: State dictionary of the model. 
                   If None, the state is initialized, the model starts from an empty initial state.
        
        Returns:
            logits: Logits tensor of shape [B, S, V].
            Tuple of logits and state: State dictionary of the model, if `return_last_states` is True.
        """

        assert x.ndim == 2, f"Input must have shape [B, S], got {x.shape}"
        B, S = x.shape

        x = self.embedding(x)

        x, state = self.backbone(x, state)

        logits = self.lm_head(x)
        logits_capped = soft_cap(logits, self.config.output_logit_soft_cap)
        if self.config.return_last_states:
            return logits_capped, state
        else:
            return logits_capped

    def generate(
        self,
        prefill_tokens: torch.Tensor,
        max_length: int,
        sampling_type: str = "greedy",
        state: mLSTMStateType | None = None,
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        """Generate tokens from the model.

        Args:
            prefill_tokens: Tensor of shape [B, S] with the prefill tokens.
            max_length: Maximum length of the generated sequence.
            sampling_type: Sampling type to use, e.g. 'greedy'.
            state: State dictionary of the model. 
                   If None, the state is initialized, the model starts from an empty initial state.
        
        Returns:
            tokens: Generated tokens tensor of shape [B, S].
            state: State dictionary of the model after the last generation step.
        """
        sampling_fn = get_sampling_fn(sampling_type)
        tokens, state = generate_tokens(
            llm_forward=self.forward,
            prefill_tokens=prefill_tokens,
            max_length=max_length,
            token_sample_fn=sampling_fn,
            state=state,
            device=str(self.embedding.weight.device),
        )
        return tokens, state

    @torch.jit.export
    def forward_with_state(
        self,
        x: torch.Tensor,
        state: TSStateType,
    ) -> Tuple[torch.Tensor, TSStateType]:
        """TorchScript-friendly forward with typed list state.

        Args:
            x: Input ids [B, S]
            state: List of optional (c,n,m) tuples, length = num_blocks
        Returns:
            logits and updated typed state list
        """
        assert x.ndim == 2, f"Input must have shape [B, S], got {x.shape}"
        x = self.embedding(x)
        x, state = self.backbone.forward_with_state(x, state)
        logits = self.lm_head(x)
        logits_capped = soft_cap(logits, self.config.output_logit_soft_cap)
        return logits_capped, state

    @torch.jit.export
    def generate_greedy(
        self,
        prefill_tokens: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, TSStateType]:
        """Scripted greedy generation (TorchScript-friendly).

        Returns generated tokens and final typed state list.
        """
        device = self.embedding.weight.device
        tokens = prefill_tokens.to(device)
        # initialize typed state list
        state: TSStateType = [None for _ in range(len(self.backbone.blocks))]

        # Prefill pass (consume input)
        logits, state = self.forward_with_state(tokens, state)

        B = tokens.size(0)
        # Greedy decode loop
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = [last_token]
        for _ in range(max_length - 1 if max_length > 0 else 0):
            logits, state = self.forward_with_state(last_token, state)
            last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out.append(last_token)

        generated = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        return generated, state


class xLSTMBlockStack(nn.Module):
    """Block stack for xLSTMTorch."""
    config_class = xLSTMTorchConfig

    def __init__(self, config: xLSTMTorchConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList(
            [mLSTMBlock(config, block_idx=i) for i in range(config.num_blocks)]
        )

        if self.config.add_out_norm:
            self.out_norm = RMSNorm(
                num_features=config.embedding_dim,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
        else:
            self.out_norm = nn.Identity()

    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        if state is None:
            state = {i: None for i in range(len(self.blocks))}

        for i, block in enumerate(self.blocks):
            block_state = state[i]
            x, block_state_new = block(x, block_state)

            if block_state is None:
                state[i] = block_state_new
            else:
                # layer state is a tuple of three tensors: c, n, m
                # we update the state in place in order to avoid creating new tensors
                for state_idx in range(len(block_state)):
                    state[i][state_idx].copy_(block_state_new[state_idx])

        x = self.out_norm(x)

        return x, state

    @torch.jit.export
    def forward_with_state(
        self, x: torch.Tensor, state: TSStateType
    ) -> Tuple[torch.Tensor, TSStateType]:
        # Ensure correct length
        num_blocks = len(self.blocks)
        if len(state) != num_blocks:
            # Create a typed list with proper length
            state = [None for _ in range(num_blocks)]

        new_state: TSStateType = [None for _ in range(num_blocks)]
        for i in range(num_blocks):
            block = self.blocks[i]
            block_state = state[i]
            x, block_state_new = block(x, block_state)
            new_state[i] = block_state_new

        x = self.out_norm(x)
        return x, new_state


class FeedForward(nn.Module):
    def __init__(self, config: xLSTMTorchConfig):
        super().__init__()
        self.config = config

        self.up_proj_dim = round_up_to_next_multiple_of(
            config.embedding_dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )

        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
            self.proj_up = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(
                in_features=config.embedding_dim,
                out_features=2 * self.up_proj_dim,
                bias=self.config.use_bias,
            )

        self.proj_down = nn.Linear(
            in_features=self.up_proj_dim,
            out_features=config.embedding_dim,
            bias=self.config.use_bias,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y


from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig


class mLSTMBlock(nn.Module):
    def __init__(self, config: xLSTMTorchConfig, block_idx: int):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.norm_mlstm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.mlstm_layer = mLSTMLayer(
            mLSTMLayerConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                norm_reduction_force_float32=config.norm_reduction_force_float32,
                qk_dim_factor=config.qk_dim_factor,
                v_dim_factor=config.v_dim_factor,
                gate_soft_cap=config.gate_soft_cap,
                weight_mode=config.weight_mode,
                mlstm_backend=mLSTMBackendConfig(
                    chunkwise_kernel=config.chunkwise_kernel,
                    sequence_kernel=config.sequence_kernel,
                    step_kernel=config.step_kernel,
                    mode=config.mode,
                    chunk_size=config.chunk_size,
                    return_last_states=config.return_last_states,
                    autocast_kernel_dtype=config.autocast_kernel_dtype,
                    eps=config.eps,
                    inference_state_dtype=config.inference_state_dtype,
                    runtime_opts=config.runtime_opts,
                ),
            )
        )
        # Decide whether to include FFN for this block based on config.ffn_blocks
        include_ffn = True
        if self.config.ffn_blocks is not None:
            include_ffn = self.block_idx in set(self.config.ffn_blocks)

        if include_ffn:
            self.norm_ffn = RMSNorm(
                num_features=config.embedding_dim,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
            self.ffn = FeedForward(config)
        else:
            self.norm_ffn = None
            self.ffn = None

    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm

        if self.ffn is not None:
            x_ffn = self.norm_ffn(x)
            x_ffn = self.ffn(x_ffn)
            x = x + x_ffn

        return x, state
