import mlx.core as mx
import mlx.nn as nn

from .components.mlstm.block import mLSTMBlock
from .components.slstm.block import sLSTMBlock

from .components.util import repeat

class xLSTMSolaceMLX(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        signature,
        inp_dim,
        head_dim,
        head_num,
        p_factor,
        ker_size,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hid_dim = head_dim * head_num

        self.embedding = nn.Embedding(
            vocab_size, 
            inp_dim, 
        )

        m_factor, s_factor = p_factor
        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num

        self.blocks = [
            mLSTMBlock(inp_dim, head_dim, head_num, m_factor, ker_size) if w else 
            sLSTMBlock(inp_dim, head_dim, head_num, s_factor, ker_size)
            for w, _ in zip(repeat(which), range(num_layers))
        ]

        self.head = nn.Linear(inp_dim, vocab_size)
        
    def __call__(self, tok, hid=None, batch_first: bool = True):
        tok = mx.atleast_2d(tok)
        seq = self.embedding(tok)             # (B, S, D) if batch_first

        if not batch_first:
            # Convert to time-major (S, B, D)
            B, S, D = seq.shape
            seq = mx.transpose(seq, (1, 0, 2))
        else:
            # Already batch-first; iterate over time
            B, S, D = seq.shape
            seq = mx.transpose(seq, (1, 0, 2))  # to (S, B, D) for step loop
        if hid is None:
            hid = [l.init_hidden(B) for l in self.blocks]

        out = []
        for inp in seq:                        # inp: (B, D)
            for i, lstm in enumerate(self.blocks):
                inp, hid[i] = lstm(inp, hid[i])
            
            out.append(inp)
        # Reconstruct (B, S, D)
        out = mx.stack(out, axis=0)           # (S, B, D)
        out = mx.transpose(out, (1, 0, 2))    # (B, S, D)
        out = self.head(out)

        return out, hid
