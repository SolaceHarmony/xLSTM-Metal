"""
Stateful MIL builder for full xLSTM decode step (LayerNorm + mLSTM/sLSTM + FFN),
with native Core ML states updated in-place (read_state/coreml_update_state).

This builds one MIL Program that unrolls L blocks. The block schedule (mLSTM=0,
sLSTM=1) is baked in at build time via the `block_types` argument.

Notes
- LayerNorm is used everywhere (upstream behavior); no RMSNorm.
- A small causal conv ring buffer per block is used at decode; implemented with
  MIL ops and maintained as a state tensor.
- Weights are expressed as TensorSpecs here for clarity; a separate “freeze”
  pass can replace them with MIL constants from trained weights for a fully
  self‑contained MLProgram.
"""

from typing import List

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types


def _sigmoid(x):
    return mb.sigmoid(x=x)


def _silu(x):
    return mb.mul(x=x, y=mb.sigmoid(x=x))


def _layer_norm(x, gamma, beta, eps=1e-5):
    return mb.layer_norm(x=x, gamma=gamma, beta=beta, epsilon=eps, axes=[-1])


def _soft_cap(x, cap):
    return mb.mul(x=cap, y=mb.tanh(x=mb.real_div(x=x, y=cap)))


def _linear(x, W, b):
    y = mb.matmul(x=x, y=W)
    if b is not None:
        y = mb.add(x=y, y=b)
    return y


def _causal_conv_step(x, conv_state, conv_w):
    """Depthwise 1D causal conv over time axis with a shared kernel per feature.

    Shapes:
      x:          (B, D)
      conv_state: (B, K-1, D)
      conv_w:     (K,)  (shared over D)

    Returns:
      y_conv:     (B, D)
      next_state: (B, K-1, D)
    """
    # Form window: concat(prev, x) over time tap axis -> (B, K, D)
    x_unsq = mb.expand_dims(x=x, axes=[1])  # (B,1,D)
    window = mb.concat(values=[conv_state, x_unsq], axis=1)

    # Compute y_conv = sum_t (conv_w[t] * window[:, t, :])
    K = mb.shape(x=conv_w)
    # Broadcast conv_w to (K, 1) then (B,K,D)
    w_unsq = mb.expand_dims(x=conv_w, axes=[-1])
    w_brd = mb.mul(x=w_unsq, y=mb.const(val=1.0))  # (K,1)
    w_brd = mb.expand_dims(x=w_brd, axes=[0])      # (1,K,1)
    y_scaled = mb.mul(x=window, y=w_brd)           # (B,K,D)
    y_conv = mb.reduce_sum(x=y_scaled, axes=[1])   # (B,D)

    # Next state: drop oldest tap, keep last K-1 taps (excluding current x)
    # next_state = window[:, 1:, :]
    next_state = mb.slice_by_index(x=window, begin=[0, 1, 0], end=[0, 0, 0], 
                                   begin_mask=[True, False, True], end_mask=[True, True, True])
    return y_conv, next_state


def build_stateful_xlstm_decode_program(*,
                                        V: int,
                                        D: int,
                                        L: int,
                                        NH: int,
                                        DHQK: int,
                                        DHV: int,
                                        K: int,
                                        block_types: List[int],
                                        eps: float = 1e-6,
                                        gate_soft_cap: float = 15.0):
    assert len(block_types) == L, "block_types must have length L"

    B = mb.get_new_symbol()

    tok_id = mb.TensorSpec((B, 1), dtype=types.int32)
    embed_W = mb.TensorSpec((V, D), dtype=types.fp32)
    lm_W = mb.TensorSpec((D, V), dtype=types.fp32)

    # Per-block params and states
    input_specs = [tok_id, embed_W, lm_W]

    ln_mlstm_gamma = []
    ln_mlstm_beta = []
    ln_ffn_gamma = []
    ln_ffn_beta = []
    conv_w = []

    Wq = []; bq = []
    Wk = []; bk = []
    Wv = []; bv = []
    Wz = []; bz = []
    Wi = []; bi = []
    Wf = []; bf = []
    Wo = []; bo = []
    Wout = []; bout = []
    Wup_gate = []; bup_gate = []
    Wup = []; bup = []
    Wdown = []; bdown = []

    # States: conv_state_i, c_i, n_i, m_i
    state_specs = []

    for i in range(L):
        ln_mlstm_gamma.append(mb.TensorSpec((D,), dtype=types.fp32))
        ln_mlstm_beta.append(mb.TensorSpec((D,), dtype=types.fp32))
        ln_ffn_gamma.append(mb.TensorSpec((D,), dtype=types.fp32))
        ln_ffn_beta.append(mb.TensorSpec((D,), dtype=types.fp32))
        conv_w.append(mb.TensorSpec((K,), dtype=types.fp32))

        Wq.append(mb.TensorSpec((D, NH*DHQK), dtype=types.fp32))
        bq.append(mb.TensorSpec((NH*DHQK,), dtype=types.fp32))
        Wk.append(mb.TensorSpec((D, NH*DHQK), dtype=types.fp32))
        bk.append(mb.TensorSpec((NH*DHQK,), dtype=types.fp32))
        Wv.append(mb.TensorSpec((D, NH*DHV), dtype=types.fp32))
        bv.append(mb.TensorSpec((NH*DHV,), dtype=types.fp32))
        # sLSTM z path uses separate projection (do not alias Wv)
        Wz.append(mb.TensorSpec((D, NH*DHV), dtype=types.fp32))
        bz.append(mb.TensorSpec((NH*DHV,), dtype=types.fp32))
        Wi.append(mb.TensorSpec((D, NH), dtype=types.fp32))
        bi.append(mb.TensorSpec((NH,), dtype=types.fp32))
        Wf.append(mb.TensorSpec((D, NH), dtype=types.fp32))
        bf.append(mb.TensorSpec((NH,), dtype=types.fp32))
        Wo.append(mb.TensorSpec((D, NH*DHV), dtype=types.fp32))
        bo.append(mb.TensorSpec((NH*DHV,), dtype=types.fp32))
        Wout.append(mb.TensorSpec((NH*DHV, D), dtype=types.fp32))
        bout.append(mb.TensorSpec((D,), dtype=types.fp32))

        Wup_gate.append(mb.TensorSpec((D, D), dtype=types.fp32))
        bup_gate.append(mb.TensorSpec((D,), dtype=types.fp32))
        Wup.append(mb.TensorSpec((D, D), dtype=types.fp32))
        bup.append(mb.TensorSpec((D,), dtype=types.fp32))
        Wdown.append(mb.TensorSpec((D, D), dtype=types.fp32))
        bdown.append(mb.TensorSpec((D,), dtype=types.fp32))

        input_specs += [ln_mlstm_gamma[i], ln_mlstm_beta[i], ln_ffn_gamma[i], ln_ffn_beta[i], conv_w[i],
                        Wq[i], bq[i], Wk[i], bk[i], Wv[i], bv[i], Wz[i], bz[i], Wi[i], bi[i], Wf[i], bf[i],
                        Wo[i], bo[i], Wout[i], bout[i], Wup_gate[i], bup_gate[i], Wup[i], bup[i], Wdown[i], bdown[i]]

        # States (native MIL): conv_state_i (B,K-1,D), c_i (B,NH,DHQK,DHV), n_i (B,NH,DHQK), m_i (B,NH,1)
        state_specs += [
            mb.StateTensorSpec((B, K-1, D), dtype=types.fp32),
            mb.StateTensorSpec((B, NH, DHQK, DHV), dtype=types.fp32),
            mb.StateTensorSpec((B, NH, DHQK), dtype=types.fp32),
            mb.StateTensorSpec((B, NH, 1), dtype=types.fp32),
        ]

    @mb.program(input_specs=input_specs + state_specs)
    def prog(
        tok_id,
        embed_W, lm_W,
        *args
    ):
        # unpack args into per-block params and states
        idx = 0
        ln_ml_g = []; ln_ml_b = []; ln_ff_g = []; ln_ff_b = []; cw = []
        Wq_l=[]; bq_l=[]; Wk_l=[]; bk_l=[]; Wv_l=[]; bv_l=[]; Wi_l=[]; bi_l=[]; Wf_l=[]; bf_l=[]; Wo_l=[]; bo_l=[]; Wout_l=[]; bout_l=[]
        Wupg_l=[]; bupg_l=[]; Wup_l=[]; bup_l=[]; Wdown_l=[]; bdown_l=[]
        for i in range(L):
            ln_ml_g.append(args[idx]); idx+=1
            ln_ml_b.append(args[idx]); idx+=1
            ln_ff_g.append(args[idx]); idx+=1
            ln_ff_b.append(args[idx]); idx+=1
            cw.append(args[idx]); idx+=1
            Wq_l.append(args[idx]); idx+=1
            bq_l.append(args[idx]); idx+=1
            Wk_l.append(args[idx]); idx+=1
            bk_l.append(args[idx]); idx+=1
            Wv_l.append(args[idx]); idx+=1
            bv_l.append(args[idx]); idx+=1
            Wz_l.append(args[idx]); idx+=1
            bz_l.append(args[idx]); idx+=1
            Wi_l.append(args[idx]); idx+=1
            bi_l.append(args[idx]); idx+=1
            Wf_l.append(args[idx]); idx+=1
            bf_l.append(args[idx]); idx+=1
            Wo_l.append(args[idx]); idx+=1
            bo_l.append(args[idx]); idx+=1
            Wout_l.append(args[idx]); idx+=1
            bout_l.append(args[idx]); idx+=1
            Wupg_l.append(args[idx]); idx+=1
            bupg_l.append(args[idx]); idx+=1
            Wup_l.append(args[idx]); idx+=1
            bup_l.append(args[idx]); idx+=1
            Wdown_l.append(args[idx]); idx+=1
            bdown_l.append(args[idx]); idx+=1

        conv_states = []
        c_states = []
        n_states = []
        m_states = []
        for i in range(L):
            conv_states.append(args[idx]); idx+=1
            c_states.append(args[idx]); idx+=1
            n_states.append(args[idx]); idx+=1
            m_states.append(args[idx]); idx+=1

        # Embedding lookup
        x = mb.gather(x=embed_W, indices=tok_id, axis=0)
        x = mb.squeeze(x=x, axes=[1])  # (B,D)

        cap = mb.const(val=gate_soft_cap)
        inv_sqrt_d = mb.const(val=(DHQK ** -0.5))

        for i in range(L):
            # Read states
            cs = mb.read_state(input=conv_states[i])
            c = mb.read_state(input=c_states[i])
            n = mb.read_state(input=n_states[i])
            m = mb.read_state(input=m_states[i])

            # Pre-norm
            x_norm = _layer_norm(x, ln_ml_g[i], ln_ml_b[i], eps)

            # Causal conv prep
            y_conv, cs_next = _causal_conv_step(x_norm, cs, cw[i])

            if block_types[i] == 0:
                # mLSTM block
                q_flat = _linear(y_conv, Wq_l[i], bq_l[i])
                k_flat = _linear(y_conv, Wk_l[i], bk_l[i])
                v_flat = _linear(x_norm, Wv_l[i], bv_l[i])
                i_raw = _soft_cap(_linear(x_norm, Wi_l[i], bi_l[i]), cap)
                f_raw = _soft_cap(_linear(x_norm, Wf_l[i], bf_l[i]), cap)
                o_flat = _linear(x_norm, Wo_l[i], bo_l[i])

                q = mb.reshape(x=q_flat, shape=[B, NH, DHQK])
                k = mb.reshape(x=k_flat, shape=[B, NH, DHQK])
                v = mb.reshape(x=v_flat, shape=[B, NH, DHV])
                i_p = mb.reshape(x=i_raw, shape=[B, NH, 1])
                f_p = mb.reshape(x=f_raw, shape=[B, NH, 1])

                # mLSTM recurrence
                f_neg = mb.neg(x=f_p)
                f_exp = mb.exp(x=f_neg)
                f_one = mb.add(x=f_exp, y=1.0)
                f_log = mb.log(x=f_one)
                f_logsig = mb.neg(x=f_log)
                fpm = mb.add(x=f_logsig, y=m)
                m_new = mb.maximum(x=fpm, y=i_p)
                fm2 = mb.add(x=f_logsig, y=m)
                F_act = mb.exp(x=mb.sub(x=fm2, y=m_new))
                I_act = mb.exp(x=mb.sub(x=i_p, y=m_new))
                q_s = mb.mul(x=q, y=inv_sqrt_d)
                k_e = mb.expand_dims(x=k, axes=[-1])
                v_e = mb.expand_dims(x=mb.expand_dims(x=v, axes=[-2]), axes=[])
                kv = mb.matmul(x=k_e, y=v_e)
                F_b = mb.expand_dims(x=mb.expand_dims(x=F_act, axes=[-1]), axes=[-1])
                I_b = mb.expand_dims(x=mb.expand_dims(x=I_act, axes=[-1]), axes=[-1])
                c_new = mb.add(x=mb.mul(x=F_b, y=c), y=mb.mul(x=I_b, y=kv))
                n_new = mb.add(x=mb.mul(x=F_act, y=n), y=mb.mul(x=I_act, y=k))
                q_s_e = mb.expand_dims(x=q_s, axes=[-2])
                h_num = mb.squeeze(x=mb.matmul(x=q_s_e, y=c_new), axes=[-2])
                qn = mb.squeeze(x=mb.matmul(x=q_s_e, y=mb.expand_dims(x=n_new, axes=[-1])), axes=[-1, -2])
                max_val = mb.exp(x=mb.neg(x=m_new))
                h_denom = mb.add(x=mb.maximum(x=mb.abs(x=qn), y=max_val), y=eps)
                h_heads = mb.real_div(x=h_num, y=h_denom)
                merged = mb.reshape(x=h_heads, shape=[B, NH*DHV])
                h_norm = mb.layer_norm(x=merged, epsilon=eps, axes=[-1])
                o_gate = _sigmoid(o_flat)
                out = _linear(mb.mul(x=h_norm, y=o_gate), Wout_l[i], bout_l[i])
                x = mb.add(x=x, y=out)
            else:
                # sLSTM block (vanilla)
                # Projections: use y_conv for i/f, x_norm for z/o
                i_raw = _linear(y_conv, Wi_l[i], bi_l[i])
                f_raw = _linear(y_conv, Wf_l[i], bf_l[i])
                z_raw = _linear(x_norm, Wz_l[i], bz_l[i])  # dedicated z projection
                o_raw = _linear(x_norm, Wo_l[i], bo_l[i])

                f_neg = mb.neg(x=f_raw)
                f_exp = mb.exp(x=f_neg)
                f_one = mb.add(x=f_exp, y=1.0)
                f_log = mb.log(x=f_one)
                logfplusm = mb.add(x=m, y=f_log)
                m_new = mb.maximum(x=i_raw, y=logfplusm)
                igate = mb.minimum(x=mb.exp(x=mb.sub(x=i_raw, y=m_new)), y=1.0)
                fgate = mb.minimum(x=mb.exp(x=mb.sub(x=logfplusm, y=m_new)), y=1.0)
                c_new = mb.add(x=mb.mul(x=fgate, y=c), y=mb.mul(x=igate, y=mb.tanh(x=z_raw)))
                n_new = mb.add(x=mb.mul(x=fgate, y=n), y=igate)
                y_new = mb.mul(x=_sigmoid(o_raw), y=mb.real_div(x=c_new, y=n_new))
                merged = mb.reshape(x=y_new, shape=[B, NH*DHV])
                h_norm = mb.layer_norm(x=merged, epsilon=eps, axes=[-1])
                out = _linear(h_norm, Wout_l[i], bout_l[i])
                x = mb.add(x=x, y=out)

            # FFN
            x_ffn = _layer_norm(x, ln_ff_g[i], ln_ff_b[i], eps)
            up_gate = _linear(x_ffn, Wupg_l[i], bupg_l[i])
            up_z = _linear(x_ffn, Wup_l[i], bup_l[i])
            ff = _linear(mb.mul(x=_silu(up_gate), y=up_z), Wdown_l[i], bdown_l[i])
            x = mb.add(x=x, y=ff)

            # Update states
            mb.coreml_update_state(state=conv_states[i], value=cs_next)
            mb.coreml_update_state(state=c_states[i], value=c_new)
            mb.coreml_update_state(state=n_states[i], value=n_new)
            mb.coreml_update_state(state=m_states[i], value=m_new)

        logits = mb.matmul(x=x, y=lm_W)
        return logits

    return prog
