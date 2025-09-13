"""
xLSTM Solace Large Model Configuration for Apple Silicon.

Based on official xlstm.xlstm_large.model but adapted for Apple Metal acceleration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class xLSTMSolaceLargeConfig:
    """Configuration for xLSTM Solace Large model optimized for Apple Silicon."""
    
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

    # Apple Metal backend settings (instead of Triton)
    chunkwise_kernel: str = "chunkwise--metal_autograd"
    """Kernel to use for chunkwise parallel processing - Apple Metal optimized.
    Available Apple kernels: 
    - 'chunkwise--metal_autograd' (pure Metal acceleration)
    - 'chunkwise--queued_compiled_steps' (Metal with queued compilation)
    - 'chunkwise--ray_compiled_steps' (Metal with Ray compilation)
    """
    sequence_kernel: str = "native_sequence__metal"
    """The sequence kernel to use for processing sequences step-by-step - Apple Metal optimized.
    Uses Metal acceleration instead of Triton for Apple Silicon.
    """
    step_kernel: str = "metal"
    """The step kernel to use for processing a single step - Apple Metal optimized.
    Uses compiled Metal kernels for generation in inference mode.
    """
    mode: str = "inference"
    """The mode of operation for the backend. Default to inference for Apple optimized usage.
    Available modes are 'train', 'train_with_padding', 'inference'.
    'inference' works with arbitrary sequence lengths, and does not support training. 
    It calls a sequence of different kernels to process the sequence.
    'train_with_padding' pads the input to multiples of `chunk_size`.
    """
    chunk_size: int = 64
    """The chunk size of the chunkwise kernel.
    If `mode` is 'train_with_padding', the inputs are padded to multiples of this size.
    """
    return_last_states: bool = True
    """Whether to return the last states of the sequence.
    Default True for Apple version to enable proper state tracking.
    """
    autocast_kernel_dtype: str = "bfloat16"
    """The dtype to use for autocast behavior in the kernel.
    If autocast is enabled all inputs are cast to this dtype before the kernel is called.
    """
    eps: float = 1e-6
    """Epsilon value for numerical stability in the kernel."""
    inference_state_dtype: str = "float32"
    """The dtype to use for the state tensors in inference mode."""
    
    # feedforward (same as official)
    ffn_proj_factor: float = 2.6667
    """The factor to determine the dimension of the intermediate projection in the feedforward layer."""
    ffn_round_up_to_multiple_of: int = 64
    """Round the intermediate projection dimension to the next multiple of this value."""
    
    # capping (same as official)
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""
    output_logit_soft_cap: float = 30.0
    """Soft cap value for the output logits."""

    weight_mode: str = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and gates.
    'fused' is beneficial in inference settings.
    """
    
    # Apple Silicon specific optimizations
    force_mps_device: bool = True
    """Force MPS device usage for Apple Silicon acceleration."""
    validate_metal_kernels: bool = True
    """Validate that only Metal kernels are used (no CPU fallbacks)."""
