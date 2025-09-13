"""
Multiprocessing-based Metal kernel implementation.

Replaces Ray with Python's built-in multiprocessing module to avoid
massive memory leaks and GPU resource holding.
"""

import multiprocessing as mp
import os
import time
import math
from typing import Tuple

import torch

from ...recurrent.metal.compiled import mlstm_recurrent_step__metal


def _process_chunk_worker(args):
    """Worker function for multiprocessing pool - processes a single chunk."""
    (
        q_chunk, k_chunk, v_chunk, i_chunk, f_chunk,
        c_initial, n_initial, m_initial,
        chunk_idx, eps
    ) = args
    
    # Process this chunk using Metal
    B, NH, chunk_size, DHQK = q_chunk.shape
    DHHV = v_chunk.shape[-1]
    
    # Initialize states for this chunk
    C = c_initial if c_initial is not None else torch.zeros(
        (B, NH, DHQK, DHHV), device=q_chunk.device, dtype=torch.float32
    )
    N = n_initial if n_initial is not None else torch.zeros(
        (B, NH, DHQK), device=q_chunk.device, dtype=torch.float32
    )
    M = m_initial if m_initial is not None else torch.zeros(
        (B, NH, 1), device=q_chunk.device, dtype=torch.float32
    )
    
    h_chunk = torch.zeros((B, NH, chunk_size, DHHV), device=q_chunk.device, dtype=torch.float32)
    
    # Process each timestep in this chunk
    for t in range(chunk_size):
        q_t = q_chunk[:, :, t, :].contiguous()
        k_t = k_chunk[:, :, t, :].contiguous() 
        v_t = v_chunk[:, :, t, :].contiguous()
        i_t = i_chunk[:, :, t].contiguous()
        f_t = f_chunk[:, :, t].contiguous()
        
        # Use Metal step kernel
        h_t, C, N, M = mlstm_recurrent_step__metal(
            q_t, k_t, v_t, i_t, f_t, C, N, M, eps
        )
        h_chunk[:, :, t, :] = h_t
    
    return chunk_idx, h_chunk, C, N, M


def mlstm_chunkwise__multiprocessing_metal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    chunk_size: int = 64,
    **kwargs
) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Multiprocessing-based Metal-accelerated chunkwise mLSTM kernel.
    
    Uses Python's multiprocessing instead of Ray to avoid memory leaks.
    """
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]
    device = q.device
    
    # Ensure we're on MPS device for Metal acceleration
    if device.type != 'mps':
        raise RuntimeError("Multiprocessing Metal backend requires MPS device")
    
    # Calculate number of chunks
    num_chunks = math.ceil(S / chunk_size)
    
    # For small sequences, use direct processing
    if num_chunks == 1 or S <= chunk_size:
        # Direct processing without multiprocessing overhead
        C = c_initial if c_initial is not None else torch.zeros(
            (B, NH, DHQK, DHHV), device=device, dtype=torch.float32
        )
        N = n_initial if n_initial is not None else torch.zeros(
            (B, NH, DHQK), device=device, dtype=torch.float32
        )
        M = m_initial if m_initial is not None else torch.zeros(
            (B, NH, 1), device=device, dtype=torch.float32
        )
        
        h_out = torch.zeros((B, NH, S, DHHV), device=device, dtype=torch.float32)
        
        for t in range(S):
            q_t = q[:, :, t, :].contiguous()
            k_t = k[:, :, t, :].contiguous()
            v_t = v[:, :, t, :].contiguous()
            i_t = i[:, :, t].contiguous()
            f_t = f[:, :, t].contiguous()
            
            h_t, C, N, M = mlstm_recurrent_step__metal(
                q_t, k_t, v_t, i_t, f_t, C, N, M, eps
            )
            h_out[:, :, t, :] = h_t
        
        if return_last_states:
            return h_out, (C, N, M)
        return h_out
    
    # For larger sequences, use multiprocessing with proper spawn method for macOS
    ctx = mp.get_context('spawn')  # Safe for macOS
    
    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(num_chunks):
        start_pos = chunk_idx * chunk_size
        end_pos = min(start_pos + chunk_size, S)
        actual_chunk_size = end_pos - start_pos
        
        # Extract chunk data
        q_chunk = q[:, :, start_pos:end_pos, :].contiguous()
        k_chunk = k[:, :, start_pos:end_pos, :].contiguous()
        v_chunk = v[:, :, start_pos:end_pos, :].contiguous()
        i_chunk = i[:, :, start_pos:end_pos].contiguous()
        f_chunk = f[:, :, start_pos:end_pos].contiguous()
        
        # Initial states (for first chunk) or None (computed sequentially)
        chunk_c_initial = c_initial if chunk_idx == 0 else None
        chunk_n_initial = n_initial if chunk_idx == 0 else None
        chunk_m_initial = m_initial if chunk_idx == 0 else None
        
        chunk_args.append((
            q_chunk, k_chunk, v_chunk, i_chunk, f_chunk,
            chunk_c_initial, chunk_n_initial, chunk_m_initial,
            chunk_idx, eps
        ))
    
    # Process chunks with multiprocessing
    # Use a reasonable number of processes, not too many
    max_workers = min(4, os.cpu_count() or 4)
    
    try:
        with ctx.Pool(processes=max_workers) as pool:
            results = pool.map(_process_chunk_worker, chunk_args)
    except Exception as e:
        # Fallback to sequential processing if multiprocessing fails
        print(f"Multiprocessing failed, falling back to sequential: {e}")
        results = [_process_chunk_worker(args) for args in chunk_args]
    
    # Collect results in order
    results.sort(key=lambda x: x[0])  # Sort by chunk_idx
    
    # Assemble output
    h_out = torch.zeros((B, NH, S, DHHV), device=device, dtype=torch.float32)
    final_C, final_N, final_M = None, None, None
    
    for chunk_idx, h_chunk, C, N, M in results:
        start_pos = chunk_idx * chunk_size
        end_pos = min(start_pos + chunk_size, S)
        h_out[:, :, start_pos:end_pos, :] = h_chunk
        
        # Keep the last chunk's states
        if chunk_idx == num_chunks - 1:
            final_C, final_N, final_M = C, N, M
    
    if return_last_states:
        return h_out, (final_C, final_N, final_M)
    return h_out
