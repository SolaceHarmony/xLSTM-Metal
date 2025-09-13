"""
xLSTM Solace Torch

PyTorch implementation of xLSTM with high-performance kernels for Apple Silicon and CUDA.
"""

from .models import xLSTMTorch, xLSTMTorchConfig

# Apple Metal optimized large models with official API compatibility
from .xlstm_large import xLSTMLarge, xLSTMLargeConfig

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
    "xLSTMTorch",
    "xLSTMTorchConfig",
    # Official API with Apple Metal acceleration
    "xLSTMLarge",
    "xLSTMLargeConfig", 
    "xLSTMLargeBlockStack",
    "RMSNorm",
    "MultiHeadLayerNorm",
    "LayerNorm",
    "FeedForward",
]
