from __future__ import annotations
from typing import Sequence, Tuple
from .model import xLSTMSolaceMLX

def create_xlstm_model(*,
    vocab_size: int,
    num_layers: int,
    signature: Sequence[int] | Tuple[int, ...] = (1, 1),
    inp_dim: int,
    head_dim: int,
    head_num: int,
    dropout: float = 0.0,
):
    return xLSTMSolaceMLX(
        vocab_size=vocab_size,
        num_layers=int(num_layers),
        signature=tuple(int(x) for x in signature),
        inp_dim=int(inp_dim),
        head_dim=int(head_dim),
        head_num=int(head_num),
        p_factor=(1.0, 1.0),
        ker_size=3,
    )
