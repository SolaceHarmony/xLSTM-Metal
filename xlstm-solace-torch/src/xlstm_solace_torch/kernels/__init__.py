#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""
xLSTM kernels package.

Provides various high-performance kernel implementations for xLSTM operations.
"""

__version__ = "1.0.3"

# Expose torch kernels
from . import torch

__all__ = [
    "torch",
]