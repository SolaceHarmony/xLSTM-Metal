import torch

from ..native.fwbw import (
    mlstm_parallel__native_autograd as _parallel_native_autograd,
)
from ..native_stablef.fwbw import (
    mlstm_parallel__native_stablef_autograd as _parallel_native_stablef_autograd,
)


def _parallel_native_compiled_autograd_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = True,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    # Strict device check: compiled path must run on Apple GPU (MPS)
    if q.device.type != 'mps':
        raise RuntimeError("mLSTM parallel compiled requires Apple MPS device; CPU/CUDA not supported.")
    # Delegate to existing native parallel implementation (pure PyTorch ops)
    return _parallel_native_autograd(
        q=q, k=k, v=v, i=i, f=f,
        c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
        return_last_states=return_last_states, eps=eps,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )


def _parallel_native_stablef_compiled_autograd_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = True,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    # Strict device check: compiled path must run on Apple GPU (MPS)
    if q.device.type != 'mps':
        raise RuntimeError("mLSTM parallel compiled (stablef) requires Apple MPS device; CPU/CUDA not supported.")
    # Delegate to stablef native implementation (pure PyTorch ops)
    return _parallel_native_stablef_autograd(
        q=q, k=k, v=v, i=i, f=f,
        c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
        return_last_states=return_last_states, eps=eps,
        autocast_kernel_dtype=autocast_kernel_dtype,
    )


try:
    mlstm_parallel__native_compiled_autograd = torch.compile(
        _parallel_native_compiled_autograd_eager,
        backend="inductor",
        mode="reduce-overhead",
    )
except Exception as e:
    raise RuntimeError(
        f"torch.compile failed for parallel native compiled kernel: {e}. No fallback allowed."
    )

try:
    mlstm_parallel__native_stablef_compiled_autograd = torch.compile(
        _parallel_native_stablef_compiled_autograd_eager,
        backend="inductor",
        mode="reduce-overhead",
    )
except Exception as e:
    raise RuntimeError(
        f"torch.compile failed for parallel native stablef compiled kernel: {e}. No fallback allowed."
    )
