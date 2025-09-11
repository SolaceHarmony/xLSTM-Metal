"""
xLSTM Solace Torch

PyTorch implementation of xLSTM with high-performance kernels for Apple Silicon and CUDA.
"""

from .models import xLSTMSolaceTorch, xLSTMSolaceTorchConfig
from .api import create_xlstm_model

__version__ = "0.1.0"

__all__ = [
    "xLSTMSolaceTorch",
    "xLSTMSolaceTorchConfig",
]
