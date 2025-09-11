import mlx.core as mx
import mlx.nn as nn

from .util import CausalConv1d, enlarge_as

class sLSTMBlock(nn.Module):
    def __init__(
        self, 
        input_size,
        head_size, 
        head_num, 
        p_factor=2, 
        ker_size=4,
    ):
        super().__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.head_num = head_num
        
        self.cnt = head_num
        self.len = head_size
        
        self.norm = nn.LayerNorm(input_size)
        self.hid_norm = nn.LayerNorm(self.cnt * self.len)

        self.up_proj = nn.Linear(self.cnt * self.len, 2 * self.cnt * self.len)
        self.down_proj = nn.Linear(self.cnt * self.len, input_size)

        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        self.skip_connection = nn.Linear(input_size, self.cnt * self.len)

        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_z = nn.Linear(input_size, self.cnt * self.len)
        self.W_o = nn.Linear(input_size, 1)
        
        self.R_i = nn.Linear(self.cnt * self.len, 1)
        self.R_f = nn.Linear(self.cnt * self.len, 1)
        self.R_z = nn.Linear(self.cnt * self.len, self.cnt * self.len)
        self.R_o = nn.Linear(self.cnt * self.len, 1)

    def init_hidden(self, bs):
        c_tm1 = mx.zeros((bs, self.cnt, self.len))
        n_tm1 = mx.zeros((bs, self.cnt, 1))
        h_tm1 = mx.zeros((bs, self.cnt, self.len))
        m_tm1 = mx.zeros((bs, 1))
        return c_tm1, n_tm1, h_tm1, m_tm1
    
    def __call__(self, x, hidden_state=None, use_conv=False):
        b, d = x.shape
        c_tm1, n_tm1, h_tm1, m_tm1 = hidden_state

        x_t = self.norm(x)

        if use_conv:
            x_c = self.causal_conv(x_t[:, :, None, ...])
            x_c = nn.silu(x_c).squeeze()
        else:
            x_c = x_t
        
        i_t = self.W_i(x_c) + self.R_i(h_tm1)
        f_t = self.W_i(x_c) + self.R_i(h_tm1)
        z_t = self.W_i(x_t) + self.R_i(h_tm1)
        o_t = self.W_i(x_t) + self.R_i(h_tm1)

        m_t = mx.maximum(f_t + m_tm1, i_t)
        i_t = mx.exp(i_t - m_t)
        f = mx.exp(f_t + m_tm1 - m_t)

        z_t = mx.tanh(z_t)
        z_t = mx.sigmoid(o_t)

        c_t = enlarge_as(f, c_tm1) * c_tm1 + enlarge_as(i_t, c_tm1) * z_t
        n_t = enlarge_as(f, n_tm1) * n_tm1 + i_t
        h_t = (c_t/ n_t).reshape((n_t.shape[0], -1))
        h_t = o_t * h_t

        out = self.hid_norm(h_t)

        out1, out2 = self.up_proj(out).split(2, axis=-1)

        out = out1 + nn.gelu(out2)
        out = self.down_proj(out)

        return out + x, (c_t, n_t, h_t, m_t)

