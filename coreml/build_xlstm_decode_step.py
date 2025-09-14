"""
Core ML MIL (MLProgram) builder for xLSTM decode step (mLSTM blocks + FFN).

Builds a single-token decode step that maps input token ids and per-block states
to next-token logits and updated states. Faithfully mirrors the upstream Large
variant with mLSTM + gated FFN per block (no sLSTM in this pass).

Shapes (B=batch, D=embedding_dim, NH=num_heads, DHQK, DHV):
Inputs:
  - tok_id:   (B, 1)          int32 token ids (optional; or pass embed directly)
  - embed_W:  (V, D)          embedding table (float32)
  - lm_W:     (D, V)          LM head weight (float32)
  - Per block i in [0, L):
      Norm params:
        - ln_mlstm_gamma_i: (D,), ln_mlstm_beta_i: (D,)
        - ln_ffn_gamma_i: (D,),   ln_ffn_beta_i: (D,)
      Projections:
        - Wq_i: (D, NH*DHQK), bq_i: (NH*DHQK,)
        - Wk_i: (D, NH*DHQK), bk_i: (NH*DHQK,)
        - Wv_i: (D, NH*DHV),  bv_i: (NH*DHV,)
        - Wi_i: (D, NH),      bi_i: (NH,)
        - Wf_i: (D, NH),      bf_i: (NH,)
        - Wo_i: (D, NH*DHV),  bo_i: (NH*DHV,)
        - Wout_i: (NH*DHV, D), bout_i: (D,)
      FFN:
        - Wup_gate_i: (D, Dff), bup_gate_i: (Dff,)
        - Wup_i:      (D, Dff), bup_i:      (Dff,)
        - Wdown_i:    (Dff, D), bdown_i:    (D,)
      States:
        - c_i: (B, NH, DHQK, DHV)
        - n_i: (B, NH, DHQK)
        - m_i: (B, NH, 1)

Outputs:
  - logits: (B, V)
  - Updated states per block: c_i', n_i', m_i'

Note:
  - soft_cap(x, cap) = cap * tanh(x / cap)
  - ogate = sigmoid(Linear_o(x))
  - mLSTM step mirrors the MIL step used in build_mlstm_step.py
  - LayerNorm used in place of RMSNorm for MIL fidelity

This builder expects weights as inputs for maximum flexibility; they can be
embedded as consts in a post-processing step if desired.
"""

from coremltools.converters.mil.mil import Program, Function
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol


def _linear(x, W, b):
    y = mb.matmul(x=x, y=W)
    if b is not None:
        y = mb.add(x=y, y=b)
    return y


def _layer_norm(x, gamma, beta, eps=1e-5):
    # Use MIL layer_norm
    return mb.layer_norm(x=x, gamma=gamma, beta=beta, epsilon=eps, axes=[-1])


def _soft_cap(x, cap):
    return mb.mul(x=cap, y=mb.tanh(x=mb.real_div(x=x, y=cap)))


def _sigmoid(x):
    return mb.sigmoid(x=x)


def _silu(x):
    return mb.mul(x=x, y=mb.sigmoid(x=x))


def build_xlstm_decode_step_program(*, V: int, D: int, L: int, NH: int, DHQK: int, DHV: int,
                                    gate_soft_cap: float = 15.0, eps: float = 1e-6):
    B = get_new_symbol()

    # Inputs
    tok_id = mb.placeholder(shape=[B, 1], dtype='int32', name='tok_id')
    embed_W = mb.placeholder(shape=[V, D], name='embed_W')
    lm_W = mb.placeholder(shape=[D, V], name='lm_W')

    # Gather embedding: x = embed[tok_id]
    x = mb.gather(x=embed_W, indices=tok_id, axis=0)
    x = mb.squeeze(x=x, axes=[1])  # (B, D)

    # Per-block placeholders
    ln_mlstm_gamma = []
    ln_mlstm_beta = []
    ln_ffn_gamma = []
    ln_ffn_beta = []
    Wq = []; bq = []
    Wk = []; bk = []
    Wv = []; bv = []
    Wi = []; bi = []
    Wf = []; bf = []
    Wo = []; bo = []
    Wout = []; bout = []
    Wup_gate = []; bup_gate = []
    Wup = []; bup = []
    Wdown = []; bdown = []
    c_list = []; n_list = []; m_list = []

    for i in range(L):
        ln_mlstm_gamma.append(mb.placeholder(shape=[D], name=f'ln_mlstm_gamma_{i}'))
        ln_mlstm_beta.append(mb.placeholder(shape=[D], name=f'ln_mlstm_beta_{i}'))
        ln_ffn_gamma.append(mb.placeholder(shape=[D], name=f'ln_ffn_gamma_{i}'))
        ln_ffn_beta.append(mb.placeholder(shape=[D], name=f'ln_ffn_beta_{i}'))

        Wq.append(mb.placeholder(shape=[D, NH*DHQK], name=f'Wq_{i}'))
        bq.append(mb.placeholder(shape=[NH*DHQK], name=f'bq_{i}'))
        Wk.append(mb.placeholder(shape=[D, NH*DHQK], name=f'Wk_{i}'))
        bk.append(mb.placeholder(shape=[NH*DHQK], name=f'bk_{i}'))
        Wv.append(mb.placeholder(shape=[D, NH*DHV], name=f'Wv_{i}'))
        bv.append(mb.placeholder(shape=[NH*DHV], name=f'bv_{i}'))
        Wi.append(mb.placeholder(shape=[D, NH], name=f'Wi_{i}'))
        bi.append(mb.placeholder(shape=[NH], name=f'bi_{i}'))
        Wf.append(mb.placeholder(shape=[D, NH], name=f'Wf_{i}'))
        bf.append(mb.placeholder(shape=[NH], name=f'bf_{i}'))
        Wo.append(mb.placeholder(shape=[D, NH*DHV], name=f'Wo_{i}'))
        bo.append(mb.placeholder(shape=[NH*DHV], name=f'bo_{i}'))
        Wout.append(mb.placeholder(shape=[NH*DHV, D], name=f'Wout_{i}'))
        bout.append(mb.placeholder(shape=[D], name=f'bout_{i}'))

        Wup_gate.append(mb.placeholder(shape=[D, D], name=f'Wup_gate_{i}'))
        bup_gate.append(mb.placeholder(shape=[D], name=f'bup_gate_{i}'))
        Wup.append(mb.placeholder(shape=[D, D], name=f'Wup_{i}'))
        bup.append(mb.placeholder(shape=[D], name=f'bup_{i}'))
        Wdown.append(mb.placeholder(shape=[D, D], name=f'Wdown_{i}'))
        bdown.append(mb.placeholder(shape=[D], name=f'bdown_{i}'))

        c_list.append(mb.placeholder(shape=[B, NH, DHQK, DHV], name=f'c_{i}'))
        n_list.append(mb.placeholder(shape=[B, NH, DHQK], name=f'n_{i}'))
        m_list.append(mb.placeholder(shape=[B, NH, 1], name=f'm_{i}'))

    cap = mb.const(val=gate_soft_cap)

    # Unroll blocks
    new_c = []; new_n = []; new_m = []
    for i in range(L):
        # Pre-norm
        x_norm = _layer_norm(x, ln_mlstm_gamma[i], ln_mlstm_beta[i], eps)

        # Projections
        q_flat = _linear(x_norm, Wq[i], bq[i])  # (B, NH*DHQK)
        k_flat = _linear(x_norm, Wk[i], bk[i])
        v_flat = _linear(x_norm, Wv[i], bv[i])
        i_gate = _soft_cap(_linear(x_norm, Wi[i], bi[i]), cap)  # (B, NH)
        f_gate = _soft_cap(_linear(x_norm, Wf[i], bf[i]), cap)
        o_flat = _linear(x_norm, Wo[i], bo[i])

        # Reshape to heads
        q = mb.reshape(x=q_flat, shape=[B, NH, DHQK])
        k = mb.reshape(x=k_flat, shape=[B, NH, DHQK])
        v = mb.reshape(x=v_flat, shape=[B, NH, DHV])
        i_p = mb.reshape(x=i_gate, shape=[B, NH, 1])
        f_p = mb.reshape(x=f_gate, shape=[B, NH, 1])
        o = mb.reshape(x=o_flat, shape=[B, NH, DHV])

        # mLSTM step (inline)
        f_neg = mb.neg(x=f_p)
        f_exp = mb.exp(x=f_neg)
        f_one = mb.add(x=f_exp, y=1.0)
        f_log = mb.log(x=f_one)
        f_logsig = mb.neg(x=f_log)
        fpm = mb.add(x=f_logsig, y=m_list[i])
        m_new = mb.maximum(x=fpm, y=i_p)
        fm2 = mb.add(x=f_logsig, y=m_list[i])
        fm_minus_m = mb.sub(x=fm2, y=m_new)
        F_act = mb.exp(x=fm_minus_m)
        I_act = mb.exp(x=mb.sub(x=i_p, y=m_new))
        inv_sqrt_d = mb.const(val=(DHQK ** -0.5))
        q_s = mb.mul(x=q, y=inv_sqrt_d)
        k_e = mb.expand_dims(x=k, axes=[-1])
        v_e = mb.expand_dims(x=mb.expand_dims(x=v, axes=[-2]), axes=[])
        kv = mb.matmul(x=k_e, y=v_e)
        F_b = mb.expand_dims(x=mb.expand_dims(x=F_act, axes=[-1]), axes=[-1])
        I_b = mb.expand_dims(x=mb.expand_dims(x=I_act, axes=[-1]), axes=[-1])
        c_new = mb.add(x=mb.mul(x=F_b, y=c_list[i]), y=mb.mul(x=I_b, y=kv))
        n_new = mb.add(x=mb.mul(x=F_act, y=n_list[i]), y=mb.mul(x=I_act, y=k))
        q_s_e = mb.expand_dims(x=q_s, axes=[-2])
        h_num_e = mb.matmul(x=q_s_e, y=c_new)
        h_num = mb.squeeze(x=h_num_e, axes=[-2])
        n_new_e = mb.expand_dims(x=n_new, axes=[-1])
        qn_e = mb.matmul(x=q_s_e, y=n_new_e)
        qn = mb.squeeze(x=qn_e, axes=[-1, -2])
        max_val = mb.exp(x=mb.neg(x=m_new))
        h_denom = mb.add(x=mb.maximum(x=mb.abs(x=qn), y=max_val), y=eps)
        h_heads = mb.real_div(x=h_num, y=h_denom)  # (B, NH, DHV)

        # Out path: normalize, gate, project, residual
        h_flat = mb.reshape(x=h_heads, shape=[B, NH*DHV])
        # Use LN over merged heads
        h_norm = mb.layer_norm(x=h_flat, epsilon=eps, axes=[-1])
        h_out = mb.mul(x=_sigmoid(o_flat), y=h_norm)
        y = _linear(h_out, Wout[i], bout[i])
        x = mb.add(x=x, y=y)

        # FFN
        x_ffn = _layer_norm(x, ln_ffn_gamma[i], ln_ffn_beta[i], eps)
        up_gate = _linear(x_ffn, Wup_gate[i], bup_gate[i])
        up_z = _linear(x_ffn, Wup[i], bup[i])
        ff = _linear(mb.mul(x=_silu(up_gate), y=up_z), Wdown[i], bdown[i])
        x = mb.add(x=x, y=ff)

        new_c.append(c_new)
        new_n.append(n_new)
        new_m.append(m_new)

    # LM head: logits = x @ lm_W
    logits = mb.matmul(x=x, y=lm_W)

    prog = Program()
    with Function('main') as ssa:
        # Re-bind placeholders are already captured by builder placeholders
        mb.output(outputs=[logits] + new_c + new_n + new_m)
    prog.add_function(ssa)
    return prog, ssa


if __name__ == '__main__':
    # Example: build a small decode step program (no save)
    prog, _ = build_xlstm_decode_step_program(V=256, D=128, L=2, NH=4, DHQK=32, DHV=32)
    print(prog)

