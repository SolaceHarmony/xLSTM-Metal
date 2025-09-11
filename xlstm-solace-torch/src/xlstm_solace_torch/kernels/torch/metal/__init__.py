"""
Metal-accelerated kernels for PyTorch xLSTM operations.

This module provides Metal Performance Shaders (MPS) accelerated implementations
of xLSTM operations for Apple Silicon devices.
"""

from .softcap import metal_soft_cap

__all__ = [
    "metal_soft_cap",
]