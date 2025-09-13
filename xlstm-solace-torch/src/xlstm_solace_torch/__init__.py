"""
xLSTM Solace Torch

PyTorch implementation of xLSTM with high-performance kernels for Apple Silicon and CUDA.
"""

from .models import xLSTMSolaceTorch, xLSTMSolaceTorchConfig

# Apple Metal optimized large models with official API compatibility
from .xlstm_solace_large import xLSTMSolaceLarge, xLSTMSolaceLargeConfig

# Component exports for compatibility
from .models.components import (
    RMSNorm,
    MultiHeadLayerNorm,
    LayerNorm,
    MultiHeadRMSNorm,
    soft_cap,
)

__version__ = "0.1.0"

__all__ = [
    # Internal Apple implementation
    "xLSTMSolaceTorch",
    "xLSTMSolaceTorchConfig",
    # Official API with Apple Metal acceleration
    "xLSTMLarge",
    "xLSTMLargeConfig", 
    "xLSTMLargeBlockStack",
    "RMSNorm",
    "MultiHeadLayerNorm",
    "LayerNorm",
    "FeedForward",
]
