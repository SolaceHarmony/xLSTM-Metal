"""
Core ML MIL (MLProgram) builder for a faithful mLSTM step cell.

Builds a single-step mLSTM cell as an MLProgram using coremltools MIL builder.
Inputs (per step, batch-first):
  - q:   (B, NH, DHQK)
  - k:   (B, NH, DHQK)
  - v:   (B, NH, DHV)
  - i:   (B, NH, 1)     input gate preact
  - f:   (B, NH, 1)     forget gate preact
  - c:   (B, NH, DHQK, DHV)  memory (matC_old)
  - n:   (B, NH, DHQK)       normalizer (vecN_old)
  - m:   (B, NH, 1)          log-max state (scaM_old)

Outputs:
  - h:   (B, NH, DHV)
  - c':  (B, NH, DHQK, DHV)
  - n':  (B, NH, DHQK)
  - m':  (B, NH, 1)

Formulae mirror upstream:
  f_log   = logsigmoid(f)
  m_new   = max(f_log + m_old, i)
  F_act   = exp(f_log + m_old - m_new)
  I_act   = exp(i - m_new)
  q_s     = q * (DHQK**-0.5)
  c_new   = F_act * c_old + I_act * (k @ v)  # batched outer product
  n_new   = F_act * n_old + I_act * k
  h_num   = (q_s @ c_new)                    # (B, NH, DHV)
  qn      = (q_s dot n_new)                  # (B, NH, 1)
  max_val = exp(-m_new)
  h_denom = max(abs(qn), max_val) + eps
  h       = h_num / h_denom

Note: this builder requires coremltools v6+ with MIL builder imports. It does not
attempt to run conversion here — it fails loudly if coremltools is unavailable.
"""

from coremltools.converters.mil.mil import Program, Function
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol


def build_mlstm_step_program(*, dhqk: int, dhv: int, nh: int, eps: float = 1e-6):
    """Construct an MLProgram for a single mLSTM step.

    Args:
        dhqk: dimension of q/k head-projected vectors per head
        dhv:  dimension of v per head
        nh:   number of heads
        eps:  numerical stability epsilon

    Returns:
        (prog, main): MIL Program and main Function
    """

    B = get_new_symbol()  # batch dimension symbol

    # Placeholders
    q = mb.placeholder(shape=[B, nh, dhqk], name="q")
    k = mb.placeholder(shape=[B, nh, dhqk], name="k")
    v = mb.placeholder(shape=[B, nh, dhv], name="v")
    i = mb.placeholder(shape=[B, nh, 1], name="i")
    f = mb.placeholder(shape=[B, nh, 1], name="f")
    c_old = mb.placeholder(shape=[B, nh, dhqk, dhv], name="c_old")
    n_old = mb.placeholder(shape=[B, nh, dhqk], name="n_old")
    m_old = mb.placeholder(shape=[B, nh, 1], name="m_old")

    # logsigmoid(f) = -log(1 + exp(-f))
    f_neg = mb.neg(x=f)
    f_exp = mb.exp(x=f_neg)
    f_one = mb.add(x=f_exp, y=1.0)
    f_log = mb.log(x=f_one)
    f_log = mb.neg(x=f_log)

    # m_new = max(f_log + m_old, i)
    f_plus_m = mb.add(x=f_log, y=m_old)
    m_new = mb.maximum(x=f_plus_m, y=i)

    # F_act = exp(f_log + m_old - m_new)
    f_plus_m2 = mb.add(x=f_log, y=m_old)
    fm_minus_mnew = mb.sub(x=f_plus_m2, y=m_new)
    F_act = mb.exp(x=fm_minus_mnew)

    # I_act = exp(i - m_new)
    i_minus_m = mb.sub(x=i, y=m_new)
    I_act = mb.exp(x=i_minus_m)

    # q_scaled = q * inv_sqrt_d
    inv_sqrt_d = (dhqk ** -0.5)
    q_s = mb.mul(x=q, y=inv_sqrt_d)

    # k @ v (outer product per (B,NH)) -> (B, NH, DHQK, DHV)
    k_e = mb.expand_dims(x=k, axes=[-1])          # (B, NH, DHQK, 1)
    v_e = mb.expand_dims(x=mb.expand_dims(x=v, axes=[-2]), axes=[])  # (B, NH, 1, DHV)
    kv = mb.matmul(x=k_e, y=v_e)

    # Broadcast gates to 4-D for c_new
    F_b = mb.expand_dims(x=mb.expand_dims(x=F_act, axes=[-1]), axes=[-1])  # (B,NH,1,1)
    I_b = mb.expand_dims(x=mb.expand_dims(x=I_act, axes=[-1]), axes=[-1])

    c_new = mb.add(x=mb.mul(x=F_b, y=c_old), y=mb.mul(x=I_b, y=kv))
    n_new = mb.add(x=mb.mul(x=F_act, y=n_old), y=mb.mul(x=I_act, y=k))

    # h_num = (q_s @ c_new) -> (B, NH, DHV)
    q_s_e = mb.expand_dims(x=q_s, axes=[-2])      # (B, NH, 1, DHQK)
    h_num_e = mb.matmul(x=q_s_e, y=c_new)         # (B, NH, 1, DHV)
    h_num = mb.squeeze(x=h_num_e, axes=[-2])      # (B, NH, DHV)

    # qn_dot = (q_s dot n_new) -> (B, NH, 1)
    n_new_e = mb.expand_dims(x=n_new, axes=[-1])  # (B, NH, DHQK, 1)
    qn_e = mb.matmul(x=q_s_e, y=n_new_e)          # (B, NH, 1, 1)
    qn = mb.squeeze(x=qn_e, axes=[-1, -2])        # (B, NH, 1)

    # h_denom = max(abs(qn), exp(-m_new)) + eps
    m_neg = mb.neg(x=m_new)
    max_val = mb.exp(x=m_neg)
    qn_abs = mb.abs(x=qn)
    denom0 = mb.maximum(x=qn_abs, y=max_val)
    h_denom = mb.add(x=denom0, y=eps)

    h = mb.real_div(x=h_num, y=h_denom)

    prog = Program()
    with Function("main") as ssa:
        # Re-bind in the function scope
        q_f = mb.placeholder(shape=q.shape, name=q.name)
        k_f = mb.placeholder(shape=k.shape, name=k.name)
        v_f = mb.placeholder(shape=v.shape, name=v.name)
        i_f = mb.placeholder(shape=i.shape, name=i.name)
        f_f = mb.placeholder(shape=f.shape, name=f.name)
        c_f = mb.placeholder(shape=c_old.shape, name=c_old.name)
        n_f = mb.placeholder(shape=n_old.shape, name=n_old.name)
        m_f = mb.placeholder(shape=m_old.shape, name=m_old.name)

        # Recompute using builder inside function scope referencing *f variables
        f_neg_f = mb.neg(x=f_f)
        f_exp_f = mb.exp(x=f_neg_f)
        f_one_f = mb.add(x=f_exp_f, y=1.0)
        f_log_f = mb.neg(x=mb.log(x=f_one_f))

        fpm_f = mb.add(x=f_log_f, y=m_f)
        m_new_f = mb.maximum(x=fpm_f, y=i_f)

        fpm2_f = mb.add(x=f_log_f, y=m_f)
        fm_minus_mnew_f = mb.sub(x=fpm2_f, y=m_new_f)
        F_act_f = mb.exp(x=fm_minus_mnew_f)

        i_minus_m_f = mb.sub(x=i_f, y=m_new_f)
        I_act_f = mb.exp(x=i_minus_m_f)

        q_s_f = mb.mul(x=q_f, y=inv_sqrt_d)

        k_e_f = mb.expand_dims(x=k_f, axes=[-1])
        v_e_f = mb.expand_dims(x=mb.expand_dims(x=v_f, axes=[-2]), axes=[])
        kv_f = mb.matmul(x=k_e_f, y=v_e_f)

        F_b_f = mb.expand_dims(x=mb.expand_dims(x=F_act_f, axes=[-1]), axes=[-1])
        I_b_f = mb.expand_dims(x=mb.expand_dims(x=I_act_f, axes=[-1]), axes=[-1])

        c_new_f = mb.add(x=mb.mul(x=F_b_f, y=c_f), y=mb.mul(x=I_b_f, y=kv_f))
        n_new_f = mb.add(x=mb.mul(x=F_act_f, y=n_f), y=mb.mul(x=I_act_f, y=k_f))

        q_s_e_f = mb.expand_dims(x=q_s_f, axes=[-2])
        h_num_e_f = mb.matmul(x=q_s_e_f, y=c_new_f)
        h_num_f = mb.squeeze(x=h_num_e_f, axes=[-2])

        n_new_e_f = mb.expand_dims(x=n_new_f, axes=[-1])
        qn_e_f = mb.matmul(x=q_s_e_f, y=n_new_e_f)
        qn_f = mb.squeeze(x=qn_e_f, axes=[-1, -2])

        m_neg_f = mb.neg(x=m_new_f)
        max_val_f = mb.exp(x=m_neg_f)
        qn_abs_f = mb.abs(x=qn_f)
        denom0_f = mb.maximum(x=qn_abs_f, y=max_val_f)
        h_denom_f = mb.add(x=denom0_f, y=eps)
        h_f = mb.real_div(x=h_num_f, y=h_denom_f)

        mb.output(outputs=[h_f, c_new_f, n_new_f, m_new_f])

    prog.add_function(ssa)
    return prog, ssa


if __name__ == "__main__":
    # Example build for DHQK=512, DHV=512, NH=8
    prog, _ = build_mlstm_step_program(dhqk=512, dhv=512, nh=8)
    print(prog)

