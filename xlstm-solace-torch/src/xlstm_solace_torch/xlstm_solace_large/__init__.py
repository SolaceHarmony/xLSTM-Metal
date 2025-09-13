"""
xLSTM Solace Large - Apple Silicon optimized large models.

Provides the same API as official xlstm.xlstm_large but with Apple Metal acceleration.
"""

from .config import xLSTMLargeConfig
from .model import xLSTMLarge

__all__ = [
    "xLSTMLargeConfig",
    "xLSTMLarge",
]
