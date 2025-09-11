import mlx.core as mx
import mlx.nn as nn

from ..util import CausalConv1d, enlarge_as, clamp

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, num_layers)
        self.W_v = nn.Linear(input_size, num_layers)

        self.input_gates = nn.Linear(input_size, 1)
        self.forget_gates = nn.Linear(input_size, 1)
        self.output_gates = nn.Linear(input_size, 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        k = mx.sqrt(0.6 / (self.input_size + self.hidden_size))
        
        self.W_q.weight = mx.random.uniform(-k, k, shape=self.W_q.weight.shape)
        self.W_k.weight = mx.random.uniform(-k, k, shape=self.W_k.weight.shape)
        self.W_v.weight = mx.random.uniform(-k, k, shape=self.W_v.weight.shape)
        self.W_q.bias = mx.zeros((self.input_size, 1))
        self.W_k.bias = mx.zeros((self.input_size, 1))
        self.W_v.bias = mx.zeros((self.input_size, 1))

        for gate in [self.input_gates, self.forget_gates, self.output_gates]:
            gate.weight = mx.random.uniform(-k, k, shape=gate.weight.shape)
            gate.bias = mx.zeros(gate.bias.shape)

    def __call__(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.init_hidden()
        
        C_prev, n_prev = hidden_state
        qt = mx.matmul(self.W_q.weight, x) + self.W_q.bias
        kt = (1 / mx.sqrt(self.num_layers)) * (mx.matmul(self.W_k.weight, x) + self.W_k.bias.T)
        vt = mx.matmul(self.W_v.weight, x) + self.W_v.bias.T

        it = mx.exp(mx.matmul(self.input_gates.weight, x) + self.input_gates.bias)
        ft = mx.sigmoid(mx.matmul(self.forget_gates.weight, x) + self.forget_gates.bias)

        vt = mx.squeeze(vt)
        kt = mx.squeeze(kt)

        C = ft * C_prev + it * mx.outer(vt, kt)
        n = ft * n_prev + it * kt[:, None, ...]

        max_nqt = mx.abs(mx.matmul(n.T, qt)).max()
        max_nqt = 1.0 if 1.0 > max_nqt else max_nqt

        h_tilde = mx.matmul(C, qt) / max_nqt
        ot = mx.sigmoid(mx.matmul(self.output_gates.weight, x) + self.output_gates.bias)
        ht = ot * h_tilde

        return ht, (C, n)        

    def init_hidden(self):
        C = mx.zeros((self.num_layers, self.num_layers))
        h = mx.zeros((self.num_layers, 1))
        return C, h


class mLSTMBlock(nn.Module):
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

        hidden_size = head_num * head_size

        self.norm = nn.LayerNorm(input_size)
        self.gn = nn.GroupNorm(head_size, hidden_size)

        self.up_l_proj = nn.Linear(input_size, int(p_factor * input_size))
        self.up_r_proj = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)

        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        self.skip_connection = nn.Linear(int(p_factor * input_size), hidden_size)

        self.W_i = nn.Linear(int(p_factor * input_size), head_size)
        self.W_f = nn.Linear(int(p_factor * input_size), head_size)
        self.W_o = nn.Linear(int(p_factor * input_size), hidden_size)
        
        self.W_q = nn.Linear(int(p_factor * input_size), hidden_size)
        self.W_k = nn.Linear(int(p_factor * input_size), hidden_size)
        self.W_v = nn.Linear(int(p_factor * input_size), hidden_size)

    def __call__(self, x, hidden_state=None):
        bs = x.shape[0]
        c_tm1, n_tm1, m_tm1 = hidden_state

        x_n = self.norm(x)

        x_t = self.up_l_proj(x_n)
        r_t = self.up_r_proj(x_n)

        x_c = self.causal_conv(x_t[:, :, None, ...]) # MLX Conv1D(N,L,C)
        x_c = nn.silu(x_c).squeeze()
        x_skip = self.skip_connection(x_c)

        q = self.W_q(x_c).reshape(bs, self.head_size, -1)
        k = (self.W_k(x_c) / mx.sqrt(self.head_size)).reshape(bs, self.head_size, -1)
        v = self.W_v(x_t).reshape(bs, self.head_size, -1)

        i_t = self.W_i(x_c)
        f_t = self.W_f(x_c)
        o = mx.sigmoid(self.W_o(x_t))

        m_t = mx.maximum(f_t + m_tm1, i_t)
        i = mx.exp(i_t - m_t)
        f = mx.exp(f_t + m_tm1 - m_t)

        for h in range(self.head_num):
            v_i = v[:, :, h]
            k_i = k[:, :, h]

            c_tm1[:, h, :, :] = enlarge_as(f, c_tm1[:, h, :, :]) * c_tm1[:, h, :, :]
            kv = mx.expand_dims(v_i, -1) * mx.expand_dims(k_i, -2)  # (B, DH, 1) * (B, 1, DH) -> (B, DH, DH)
            c_tm1[:, h, :, :] += enlarge_as(i, c_tm1[:, h, :, :]) * kv

            n_tm1[:, h, :] = f * n_tm1[:, h, :]
            n_tm1[:, h, :] += i * k_i

        m_tm1 = m_t

        out = []
        for h in range(self.head_num):
            v_i = v[:, :, h]           # (B, DH)
            k_i = k[:, :, h]           # (B, DH)
            q_i = q[:, :, h]           # (B, DH)

            # Per-batch normalization term: |q · n|, lower-bounded by 1.0
            nH = mx.sum(n_tm1[:, h, :] * q_i, axis=-1)  # (B,)
            nH = mx.maximum(mx.abs(nH), mx.array(1.0, dtype=nH.dtype))  # (B,)

            # Batched matrix-vector: (B, DH, DH) @ (B, DH, 1) -> (B, DH, 1) -> (B, DH)
            scv = mx.matmul(c_tm1[:, h, :, :], mx.expand_dims(q_i, -1)).squeeze(-1)
            scv = scv / mx.expand_dims(nH, -1)
            out.append(scv)
        out = mx.concatenate(out, axis=1)
        out = self.gn(out)
        
        out = nn.silu(out) * o
        out = r_t * out

        out = self.down_proj(out) + x
        return out, (c_tm1, n_tm1, m_tm1)

    def init_hidden(self, bs: int):
        """Initialize hidden state for mLSTMBlock.

        Shapes follow the internal usage:
        - c: (B, head_num, head_size, head_size)
        - n: (B, head_num, head_size)
        - m: (B, head_size) — matches per-channel gating used in m_t computation
        """
        c = mx.zeros((bs, self.head_num, self.head_size, self.head_size))
        n = mx.zeros((bs, self.head_num, self.head_size))
        m = mx.zeros((bs, self.head_size))
        return c, n, m
