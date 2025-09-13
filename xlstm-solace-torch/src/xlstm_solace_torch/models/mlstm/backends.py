"""
Apple Silicon backends for xLSTM mLSTM layer.

This module provides the same API as official xlstm.blocks.mlstm.backends
but uses Apple Metal streaming kernels instead of Ray or Triton.
"""

import torch
from typing import Optional, Tuple
from ...kernels.torch.registry import get_mlstm_kernel


def chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    initial_C: Optional[torch.Tensor] = None,
    initial_n: Optional[torch.Tensor] = None,
    initial_m: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Official API-compatible chunkwise mLSTM with Apple Metal streaming.
    
    Uses MPS streaming instead of Ray to eliminate memory leaks while 
    providing the same API as xlstm.blocks.mlstm.backends.chunkwise_simple.
    """
    # Use streaming instead of Ray for memory efficiency
    streaming_kernel = get_mlstm_kernel("chunkwise--mps_streaming")
    
    result = streaming_kernel(
        q=queries,
        k=keys,
        v=values,
        i=igate_preact,
        f=fgate_preact,
        c_initial=initial_C,
        n_initial=initial_n,
        m_initial=initial_m,
        chunk_size=chunk_size,
        return_last_states=return_last_state,
        eps=eps,
        **kwargs
    )
    
    return result


def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: Optional[torch.Tensor] = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    Official API-compatible parallel mLSTM with Apple Metal optimization.
    
    Same API as xlstm.blocks.mlstm.backends.parallel_stabilized_simple
    but uses Metal kernels for Apple Silicon acceleration.
    """
    parallel_kernel = get_mlstm_kernel("parallel--native_stablef_autograd")
    
    result = parallel_kernel(
        q=queries,
        k=keys,
        v=values,
        i=igate_preact,
        f=fgate_preact,
        eps=eps,
        **kwargs
    )
    
    return result


# Apple Metal optimized variants for power users
def chunkwise_metal_optimized(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    initial_C: Optional[torch.Tensor] = None,
    initial_n: Optional[torch.Tensor] = None,
    initial_m: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    High-performance Metal-optimized chunkwise kernel.
    Uses queued compiled steps for maximum Metal acceleration without Ray memory leaks.
    """
    optimized_kernel = get_mlstm_kernel("chunkwise--queued_compiled_steps")
    
    result = optimized_kernel(
        q=queries,
        k=keys,
        v=values,
        i=igate_preact,
        f=fgate_preact,
        c_initial=initial_C,
        n_initial=initial_n,
        m_initial=initial_m,
        chunk_size=chunk_size,
        return_last_states=return_last_state,
        eps=eps,
        **kwargs
    )
    
    return result


def recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Official API-compatible recurrent step mLSTM with Apple Metal optimization.
    
    Same API as xlstm.blocks.mlstm.backends.recurrent_step_stabilized_simple
    but uses Metal step kernels for Apple Silicon acceleration.
    """
    # Use Metal recurrent step kernel
    step_kernel = get_mlstm_kernel("recurrent--metal")
    
    result = step_kernel(
        c=c_state,
        n=n_state,
        m=m_state,
        q=q,
        k=k,
        v=v,
        i=igate_preact,
        f=fgate_preact,
        eps=eps,
        **kwargs
    )
    
    return result


def metal_streaming_optimized(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    initial_C: Optional[torch.Tensor] = None,
    initial_n: Optional[torch.Tensor] = None,
    initial_m: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Pure MPS streaming kernel using PyTorch native MPS asynchronous execution.
    """
    streaming_kernel = get_mlstm_kernel("chunkwise--mps_streaming")
    
    result = streaming_kernel(
        q=queries,
        k=keys,
        v=values,
        i=igate_preact,
        f=fgate_preact,
        c_initial=initial_C,
        n_initial=initial_n,
        m_initial=initial_m,
        chunk_size=chunk_size,
        return_last_states=return_last_state,
        eps=eps,
        **kwargs
    )
    
    return result


def metal_parallel_optimized(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    Apple Metal-accelerated parallel mLSTM implementation.
    Uses native Metal kernels for Apple Silicon optimization.
    """
    parallel_kernel = get_mlstm_kernel("parallel--native_stablef_autograd")
    
    result = parallel_kernel(
        q=queries,
        k=keys,
        v=values,
        i=igate_preact,
        f=fgate_preact,
        eps=eps,
        **kwargs
    )
    
    return result
