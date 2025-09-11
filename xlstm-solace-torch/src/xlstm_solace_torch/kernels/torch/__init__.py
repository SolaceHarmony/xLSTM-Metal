#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""
PyTorch xLSTM kernels package.

This package provides various kernel implementations for xLSTM operations including:
- Chunkwise parallel kernels
- Fully parallel kernels  
- Recurrent kernels
- Metal-accelerated kernels
- Backend configuration and selection
"""

# Main backend configuration
from .backend_module import mLSTMBackend, mLSTMBackendConfig

# Kernel registry functions
from .registry import (
    get_available_mlstm_kernels,
    get_mlstm_kernel,
    get_available_mlstm_step_kernels,
    get_mlstm_step_kernel,
    get_available_mlstm_sequence_kernels,
    get_mlstm_sequence_kernel,
)

# Utility functions
from .kernel_wrappers import *

__all__ = [
    # Backend classes
    "mLSTMBackend",
    "mLSTMBackendConfig",
    # Registry functions
    "get_available_mlstm_kernels",
    "get_mlstm_kernel", 
    "get_available_mlstm_step_kernels",
    "get_mlstm_step_kernel",
    "get_available_mlstm_sequence_kernels", 
    "get_mlstm_sequence_kernel",
]
