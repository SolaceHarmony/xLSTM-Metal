from __future__ import annotations
from typing import Sequence, Tuple, Optional
from .models import xLSTMTorch, xLSTMTorchConfig
from .kernel_validation import validate_metal_only_kernels
from .config_loader import load_model_config, ModelConfig

def create_xlstm_model(*,
    vocab_size: int,
    num_layers: int,
    signature: Sequence[int] | Tuple[int, ...] = (1, 1),
    inp_dim: int,
    head_dim: int,
    head_num: int,
    dropout: float = 0.0,
    device: str = "mps",
    config_name: Optional[str] = None,
):
    """Create an xLSTM model with the given configuration.
    
    Args:
        vocab_size: Size of the vocabulary
        num_layers: Number of transformer blocks
        signature: Pattern of mLSTM and sLSTM blocks (e.g., (7, 1) for 7 mLSTM + 1 sLSTM)
                  Note: Current implementation uses all mLSTM blocks
        inp_dim: Input/embedding dimension
        head_dim: Dimension per attention head
        head_num: Number of attention heads  
        dropout: Dropout rate (currently not used in this config)
        device: Device to place the model on (must be 'mps' for Metal acceleration)
        config_name: Name of configuration to load (default: "default_metal")
        
    Returns:
        xLSTMTorch model instance
    """
    # Load default Metal configuration
    if config_name is None:
        config_name = "default_metal"
    
    try:
        default_config = load_model_config(config_name)
    except FileNotFoundError:
        print(f"Warning: Config '{config_name}' not found, using built-in Metal defaults")
        default_config = ModelConfig()  # Uses Metal defaults
    # Force MPS device for Metal optimization
    if device != "mps":
        import torch
        if torch.backends.mps.is_available():
            print(f"WARNING: Forcing device to 'mps' instead of '{device}' for Metal acceleration")
            device = "mps"
        else:
            raise RuntimeError("MPS not available - this model requires Metal acceleration")
    
    # Calculate qk_dim_factor based on head_dim and inp_dim
    qk_dim_factor = (head_dim * head_num) / inp_dim if inp_dim > 0 else default_config.qk_dim_factor
    
    config = xLSTMTorchConfig(
        embedding_dim=int(inp_dim),
        num_heads=int(head_num),
        num_blocks=int(num_layers),
        vocab_size=int(vocab_size),
        use_bias=default_config.use_bias,
        norm_eps=default_config.norm_eps,
        norm_reduction_force_float32=default_config.norm_reduction_force_float32,
        add_out_norm=default_config.add_out_norm,
        qk_dim_factor=qk_dim_factor,
        v_dim_factor=default_config.v_dim_factor,
        gate_soft_cap=default_config.gate_soft_cap,
        output_logit_soft_cap=default_config.output_logit_soft_cap,
        # FORCE Metal-accelerated kernels from config - NO CPU/native fallbacks
        chunkwise_kernel=default_config.chunkwise_kernel,
        sequence_kernel=default_config.sequence_kernel,
        step_kernel=default_config.step_kernel,
        mode=default_config.mode,
        chunk_size=default_config.chunk_size,
    )
    
    model = xLSTMTorch(config)
    
    # Validate Metal-only acceleration
    validate_metal_only_kernels(config)
    
    if device:
        model = model.to(device)
    return model


def create_xlstm_7b_model(*, config_name: str = "xlstm_7b_metal", device: str = "mps") -> xLSTMTorch:
    """Create xLSTM 7B model with Metal acceleration using bundled configuration.
    
    Args:
        config_name: Configuration name to load (default: "xlstm_7b_metal")
        device: Device to place model on (must be 'mps')
        
    Returns:
        xLSTMTorch model configured for 7B parameters
    """
    # Load 7B configuration
    config_dict = load_model_config(config_name)
    
    # Force MPS device
    if device != "mps":
        import torch
        if torch.backends.mps.is_available():
            print(f"WARNING: Forcing device to 'mps' instead of '{device}' for Metal acceleration")
            device = "mps"
        else:
            raise RuntimeError("MPS not available - this model requires Metal acceleration")
    
    # Create model configuration
    config = xLSTMTorchConfig(
        embedding_dim=config_dict.embedding_dim,
        num_heads=config_dict.num_heads,
        num_blocks=config_dict.num_blocks,
        vocab_size=config_dict.vocab_size,
        use_bias=config_dict.use_bias,
        norm_eps=config_dict.norm_eps,
        norm_reduction_force_float32=config_dict.norm_reduction_force_float32,
        add_out_norm=config_dict.add_out_norm,
        qk_dim_factor=config_dict.qk_dim_factor,
        v_dim_factor=config_dict.v_dim_factor,
        gate_soft_cap=config_dict.gate_soft_cap,
        output_logit_soft_cap=config_dict.output_logit_soft_cap,
        chunkwise_kernel=config_dict.chunkwise_kernel,
        sequence_kernel=config_dict.sequence_kernel,
        step_kernel=config_dict.step_kernel,
        mode=config_dict.mode,
        chunk_size=config_dict.chunk_size,
    )
    
    model = xLSTMTorch(config)
    
    # Validate Metal-only acceleration
    validate_metal_only_kernels(config)
    
    if device:
        model = model.to(device)
    return model
