from __future__ import annotations
from typing import Sequence, Tuple
from .models import xLSTMSolaceTorch, xLSTMSolaceTorchConfig

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
        device: Device to place the model on
        
    Returns:
        xLSTMSolaceTorch model instance
    """
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
        # Use default backend settings for Metal/MPS
        chunkwise_kernel="chunkwise--triton_limit_chunk",
        sequence_kernel="native_sequence__triton", 
        step_kernel="triton",
        mode="train",
        chunk_size=64,
    )
    
    model = xLSTMSolaceTorch(config)
    if device:
        model = model.to(device)
    return model
