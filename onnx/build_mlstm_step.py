"""
Build an ONNX graph for a single mLSTM step using only standard ops.

Inputs:
  q:     (B, NH, DHQK)
  k:     (B, NH, DHQK)
  v:     (B, NH, DHV)
  i:     (B, NH, 1)
  f:     (B, NH, 1)
  c_old: (B, NH, DHQK, DHV)
  n_old: (B, NH, DHQK)
  m_old: (B, NH, 1)

Outputs:
  h:     (B, NH, DHV)
  c_new: (B, NH, DHQK, DHV)
  n_new: (B, NH, DHQK)
  m_new: (B, NH, 1)

Equations mirror upstream recurrent step (see docs/research/xlstm_mad_journal.md).

Usage:
  PYTHONPATH=. python onnx/build_mlstm_step.py --dhqk 512 --dhv 512 --nh 8 --out mlstm_step.onnx
"""

import argparse
import math
import onnx
from onnx import helper, TensorProto


def make_value_info(name, elem_type, dims):
    return helper.make_tensor_value_info(name, elem_type, dims)


def make_const(name, np_type, vals, shape=None):
    import numpy as np
    arr = np.array(vals, dtype=np_type)
    if shape is not None:
        arr = arr.reshape(shape)
    return helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=onnx.numpy_helper.from_array(arr, name=name),
        name=f"Const_{name}",
    )


def build_graph(dhqk: int, dhv: int, nh: int, eps: float = 1e-6, model_opset: int = 13) -> onnx.ModelProto:
    B = 'B'
    q = make_value_info('q', TensorProto.FLOAT, [B, nh, dhqk])
    k = make_value_info('k', TensorProto.FLOAT, [B, nh, dhqk])
    v = make_value_info('v', TensorProto.FLOAT, [B, nh, dhv])
    i = make_value_info('i', TensorProto.FLOAT, [B, nh, 1])
    f = make_value_info('f', TensorProto.FLOAT, [B, nh, 1])
    c_old = make_value_info('c_old', TensorProto.FLOAT, [B, nh, dhqk, dhv])
    n_old = make_value_info('n_old', TensorProto.FLOAT, [B, nh, dhqk])
    m_old = make_value_info('m_old', TensorProto.FLOAT, [B, nh, 1])

    h = make_value_info('h', TensorProto.FLOAT, [B, nh, dhv])
    c_new = make_value_info('c_new', TensorProto.FLOAT, [B, nh, dhqk, dhv])
    n_new = make_value_info('n_new', TensorProto.FLOAT, [B, nh, dhqk])
    m_new = make_value_info('m_new', TensorProto.FLOAT, [B, nh, 1])

    nodes = []

    # Constants
    nodes.append(make_const('one_f', 'float32', [1.0]))
    nodes.append(make_const('eps_f', 'float32', [eps]))
    inv_sqrt_d = 1.0 / math.sqrt(float(dhqk))
    nodes.append(make_const('inv_sqrt_d', 'float32', [inv_sqrt_d]))

    # logsigmoid(f) = -log(1 + exp(-f))
    f_neg = helper.make_node('Neg', inputs=['f'], outputs=['f_neg'], name='Neg_f')
    f_exp = helper.make_node('Exp', inputs=['f_neg'], outputs=['f_exp'], name='Exp_fneg')
    f_plus1 = helper.make_node('Add', inputs=['f_exp', 'one_f'], outputs=['f_one'], name='Add_one')
    f_log = helper.make_node('Log', inputs=['f_one'], outputs=['f_log_abs'], name='Log')
    f_logsig = helper.make_node('Neg', inputs=['f_log_abs'], outputs=['f_logsig'], name='Neg_log')
    nodes += [f_neg, f_exp, f_plus1, f_log, f_logsig]

    # m_new = max(f_logsig + m_old, i)
    f_plus_m = helper.make_node('Add', inputs=['f_logsig', 'm_old'], outputs=['fpm'], name='Add_fm')
    mnew = helper.make_node('Max', inputs=['fpm', 'i'], outputs=['m_new'], name='Max_m')
    nodes += [f_plus_m, mnew]

    # F_act = exp(f_logsig + m_old - m_new), I_act = exp(i - m_new)
    fpm2 = helper.make_node('Add', inputs=['f_logsig', 'm_old'], outputs=['fpm2'], name='Add_fm2')
    fm_minus_m = helper.make_node('Sub', inputs=['fpm2', 'm_new'], outputs=['fm_minus_m'], name='Sub_fm_m')
    F_act = helper.make_node('Exp', inputs=['fm_minus_m'], outputs=['F_act'], name='Exp_F')
    i_minus_m = helper.make_node('Sub', inputs=['i', 'm_new'], outputs=['i_minus_m'], name='Sub_i_m')
    I_act = helper.make_node('Exp', inputs=['i_minus_m'], outputs=['I_act'], name='Exp_I')
    nodes += [fpm2, fm_minus_m, F_act, i_minus_m, I_act]

    # q_scaled = q * inv_sqrt_d
    q_s = helper.make_node('Mul', inputs=['q', 'inv_sqrt_d'], outputs=['q_s'], name='Mul_q_scale')
    nodes.append(q_s)

    # kv outer product: (B,NH,DHQK,1) x (B,NH,1,DHV) -> (B,NH,DHQK,DHV)
    k_e = helper.make_node('Unsqueeze', inputs=['k'], outputs=['k_e'], name='Unsq_k', axes=[-1])
    v_e1 = helper.make_node('Unsqueeze', inputs=['v'], outputs=['v_e1'], name='Unsq_v1', axes=[-2])
    v_e = helper.make_node('Unsqueeze', inputs=['v_e1'], outputs=['v_e'], name='Unsq_v2', axes=[])
    kv = helper.make_node('MatMul', inputs=['k_e', 'v_e'], outputs=['kv'], name='MatMul_kv')
    nodes += [k_e, v_e1, v_e, kv]

    # Broadcast F_act/I_act to (B,NH,1,1)
    F_e1 = helper.make_node('Unsqueeze', inputs=['F_act'], outputs=['F_e1'], name='Unsq_F1', axes=[-1])
    F_b = helper.make_node('Unsqueeze', inputs=['F_e1'], outputs=['F_b'], name='Unsq_F2', axes=[-1])
    I_e1 = helper.make_node('Unsqueeze', inputs=['I_act'], outputs=['I_e1'], name='Unsq_I1', axes=[-1])
    I_b = helper.make_node('Unsqueeze', inputs=['I_e1'], outputs=['I_b'], name='Unsq_I2', axes=[-1])
    nodes += [F_e1, F_b, I_e1, I_b]

    F_c = helper.make_node('Mul', inputs=['F_b', 'c_old'], outputs=['F_c'], name='Mul_Fc')
    I_kv = helper.make_node('Mul', inputs=['I_b', 'kv'], outputs=['I_kv'], name='Mul_Ikv')
    c_new_node = helper.make_node('Add', inputs=['F_c', 'I_kv'], outputs=['c_new'], name='Add_cnew')
    nodes += [F_c, I_kv, c_new_node]

    # n_new = F_act * n_old + I_act * k
    Fn = helper.make_node('Mul', inputs=['F_act', 'n_old'], outputs=['Fn'], name='Mul_Fn')
    Ik = helper.make_node('Mul', inputs=['I_act', 'k'], outputs=['Ik'], name='Mul_Ik')
    n_new_node = helper.make_node('Add', inputs=['Fn', 'Ik'], outputs=['n_new'], name='Add_nnew')
    nodes += [Fn, Ik, n_new_node]

    # h_num = (q_s @ c_new): (B,NH,1,DHQK) x (B,NH,DHQK,DHV) -> (B,NH,1,DHV) -> squeeze
    q_s_e = helper.make_node('Unsqueeze', inputs=['q_s'], outputs=['q_s_e'], name='Unsq_qs', axes=[-2])
    h_num_e = helper.make_node('MatMul', inputs=['q_s_e', 'c_new'], outputs=['h_num_e'], name='MatMul_qc')
    h_num = helper.make_node('Squeeze', inputs=['h_num_e'], outputs=['h_num'], name='Squeeze_hnum', axes=[-2])
    nodes += [q_s_e, h_num_e, h_num]

    # qn = (q_s dot n_new): (B,NH,1,DHQK) x (B,NH,DHQK,1) -> (B,NH,1,1) -> squeeze to (B,NH,1)
    n_new_e = helper.make_node('Unsqueeze', inputs=['n_new'], outputs=['n_new_e'], name='Unsq_n', axes=[-1])
    qn_e = helper.make_node('MatMul', inputs=['q_s_e', 'n_new_e'], outputs=['qn_e'], name='MatMul_qn')
    qn_sq = helper.make_node('Squeeze', inputs=['qn_e'], outputs=['qn'], name='Squeeze_qn', axes=[-1, -2])
    nodes += [n_new_e, qn_e, qn_sq]

    # denom = max(abs(qn), exp(-m_new)) + eps
    m_neg = helper.make_node('Neg', inputs=['m_new'], outputs=['m_neg'], name='Neg_m')
    max_val = helper.make_node('Exp', inputs=['m_neg'], outputs=['max_val'], name='Exp_negm')
    qn_abs = helper.make_node('Abs', inputs=['qn'], outputs=['qn_abs'], name='Abs_qn')
    denom0 = helper.make_node('Max', inputs=['qn_abs', 'max_val'], outputs=['denom0'], name='Max_den')
    h_denom = helper.make_node('Add', inputs=['denom0', 'eps_f'], outputs=['h_denom'], name='Add_eps')
    nodes += [m_neg, max_val, qn_abs, denom0, h_denom]

    # h = h_num / h_denom
    h_node = helper.make_node('Div', inputs=['h_num', 'h_denom'], outputs=['h'], name='Div_h')
    nodes.append(h_node)

    graph = helper.make_graph(nodes, 'mLSTM_step',
                              inputs=[q, k, v, i, f, c_old, n_old, m_old],
                              outputs=[h, c_new, n_new, m_new])

    model = helper.make_model(graph, producer_name='xlstm-onnx', opset_imports=[helper.make_opsetid('', model_opset)])
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dhqk', type=int, default=128)
    ap.add_argument('--dhv', type=int, default=128)
    ap.add_argument('--nh', type=int, default=4)
    ap.add_argument('--out', type=str, default='mlstm_step.onnx')
    args = ap.parse_args()
    model = build_graph(dhqk=args.dhqk, dhv=args.dhv, nh=args.nh)
    onnx.save(model, args.out)
    print(f"wrote {args.out}")

