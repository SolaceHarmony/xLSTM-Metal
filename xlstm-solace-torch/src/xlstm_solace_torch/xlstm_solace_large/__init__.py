"""
xLSTM Solace Large - Apple Silicon optimized large models.

Provides the same API as official xlstm.xlstm_large but with Apple Metal acceleration.
"""

from .config import xLSTMSolaceLargeConfig
from .model import xLSTMSolaceLarge

__all__ = [
    "xLSTMSolaceLargeConfig",
    "xLSTMSolaceLarge",
]
