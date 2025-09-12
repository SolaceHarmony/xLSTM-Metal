"""
Apple Silicon backends for xLSTM mLSTM layer.

This module provides the same API as the official xlstm.blocks.mlstm.backends
but uses Apple Metal-accelerated kernels instead of Triton.
"""

import torch
from typing import Optional, Tuple
from ...kernels.torch.registry import get_mlstm_kernel, get_mlstm_step_kernel, get_mlstm_sequence_kernel


def chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,  # B, NH, S, DH
    values: torch.Tensor,  # B, NH, S, DH
    igate_preact: torch.Tensor,  # B, NH, S
    fgate_preact: torch.Tensor,  # B, NH, S
    initial_C: Optional[torch.Tensor] = None,  # B, NH, DH, DH
    initial_n: Optional[torch.Tensor] = None,  # B, NH, DH
    initial_m: Optional[torch.Tensor] = None,  # B, NH, 1
    chunk_size: int = 64,
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Apple Metal-accelerated chunkwise mLSTM implementation.
    
    Uses Metal kernels instead of Triton for Apple Silicon optimization.
    Same API as official xlstm.blocks.mlstm.backends.chunkwise_simple.
    """
    # Use our Metal-accelerated chunkwise kernel
    metal_kernel = get_mlstm_kernel("chunkwise--metal_autograd")
    
    # Map parameter names from official API to our kernel API
    result = metal_kernel(
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
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    Apple Metal-accelerated parallel mLSTM implementation.
    
    Uses Metal kernels instead of Triton for Apple Silicon optimization.
    Same API as official xlstm.blocks.mlstm.backends.parallel_stabilized_simple.
    """
    # Use our Metal-accelerated parallel kernel
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
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Apple Metal-accelerated recurrent step mLSTM implementation.
    
    Uses Metal kernels instead of Triton for Apple Silicon optimization.
    Same API as official xlstm.blocks.mlstm.backends.recurrent_step_stabilized_simple.
    """
    # Use our Metal-accelerated step kernel
    step_kernel = get_mlstm_step_kernel("metal")
    
    # Our step kernel expects different parameter order
    result = step_kernel(
        matC_old=c_state,
        vecN_old=n_state,
        scaM_old=m_state,
        vecQ=q,
        vecK=k,
        vecV=v,
        scaI=igate_preact,
        scaF=fgate_preact,
        eps=eps,
        **kwargs
    )
    
    return result


# Alternative kernels for different performance profiles
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
    Uses ray_compiled_steps for maximum Metal acceleration.
    """
    optimized_kernel = get_mlstm_kernel("chunkwise--ray_compiled_steps")
    
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
