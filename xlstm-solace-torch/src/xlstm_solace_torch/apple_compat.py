"""
Apple Silicon compatibility layer for xLSTM.

This module provides drop-in replacements for official xlstm components
that use Apple Metal acceleration instead of Triton.
"""

from xlstm_solace_torch.models.model import xLSTMSolaceTorch, xLSTMSolaceTorchConfig


def create_apple_xlstm_config(**kwargs):
    """Create xLSTM config optimized for Apple Silicon.
    
    Uses Metal kernels instead of Triton and sets optimal defaults
    for Apple Silicon performance.
    """
    # Default to Apple Metal kernels
    apple_defaults = {
        "chunkwise_kernel": "chunkwise--metal_autograd",  # Use Metal instead of Triton
        "sequence_kernel": "native_sequence__metal",      # Use Metal instead of Triton  
        "step_kernel": "metal",                           # Use Metal instead of Triton
        "mode": "inference",                              # Default to inference mode
        "return_last_states": True,                       # Enable state tracking
    }
    
    # Merge with user provided kwargs
    config_dict = {**apple_defaults, **kwargs}
    
    return xLSTMSolaceTorchConfig(**config_dict)


def create_apple_xlstm_model(**kwargs):
    """Create xLSTM model optimized for Apple Silicon.
    
    Drop-in replacement for official xlstm that uses Metal acceleration.
    """
    config = create_apple_xlstm_config(**kwargs)
    model = xLSTMSolaceTorch(config)
    
    # Move to MPS device if available
    import torch
    if torch.backends.mps.is_available():
        model = model.to('mps')
    
    return model


# Alias for compatibility with main branch scripts
xLSTMLarge = xLSTMSolaceTorch
xLSTMLargeConfig = xLSTMSolaceTorchConfig
