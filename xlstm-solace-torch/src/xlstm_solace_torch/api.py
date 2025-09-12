from __future__ import annotations
from typing import Sequence, Tuple
from .models import xLSTMSolaceTorch, xLSTMSolaceTorchConfig
from .kernel_validation import validate_metal_only_kernels

def create_xlstm_model(*,
    vocab_size: int,
    num_layers: int,
    signature: Sequence[int] | Tuple[int, ...] = (1, 1),
    inp_dim: int,
    head_dim: int,
    head_num: int,
    dropout: float = 0.0,
    device: str = "mps",
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
        
    Returns:
        xLSTMSolaceTorch model instance
    """
    # Force MPS device for Metal optimization
    if device != "mps":
        import torch
        if torch.backends.mps.is_available():
            print(f"WARNING: Forcing device to 'mps' instead of '{device}' for Metal acceleration")
            device = "mps"
        else:
            raise RuntimeError("MPS not available - this model requires Metal acceleration")
    
    # Calculate qk_dim_factor based on head_dim and inp_dim
    qk_dim_factor = (head_dim * head_num) / inp_dim if inp_dim > 0 else 0.5
    
    config = xLSTMSolaceTorchConfig(
        embedding_dim=int(inp_dim),
        num_heads=int(head_num),
        num_blocks=int(num_layers),
        vocab_size=int(vocab_size),
        use_bias=False,
        norm_eps=1e-6,
        norm_reduction_force_float32=True,
        add_out_norm=True,
        qk_dim_factor=qk_dim_factor,
        v_dim_factor=1.0,
        gate_soft_cap=15.0,
        output_logit_soft_cap=30.0,
        # FORCE Metal-accelerated kernels - NO CPU/native fallbacks
        chunkwise_kernel="chunkwise--metal_autograd",
        sequence_kernel="native_sequence__metal", 
        step_kernel="metal",
        mode="train",
        chunk_size=64,
    )
    
    model = xLSTMSolaceTorch(config)
    
    # Validate Metal-only acceleration
    validate_metal_only_kernels(config)
    
    if device:
        model = model.to(device)
    return model
