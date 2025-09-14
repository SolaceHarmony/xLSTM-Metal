"""
Build an ONNX graph for mLSTM sequence via Scan (standard ops only).

Inputs (time-major):
  Q:      (S, B, NH, DHQK)
  K:      (S, B, NH, DHQK)
  V:      (S, B, NH, DHV)
  I:      (S, B, NH, 1)
  F:      (S, B, NH, 1)
  C_init: (B, NH, DHQK, DHV)
  N_init: (B, NH, DHQK)
  M_init: (B, NH, 1)

Outputs:
  H:      (S, B, NH, DHV)
  C_out:  (B, NH, DHQK, DHV)
  N_out:  (B, NH, DHQK)
  M_out:  (B, NH, 1)

Usage:
  PYTHONPATH=. python onnx/build_mlstm_scan.py --dhqk 128 --dhv 128 --nh 4 --out mlstm_scan.onnx
"""

import argparse
import math
import onnx
from onnx import helper, TensorProto


def make_vi(name, elem, dims):
    return helper.make_tensor_value_info(name, elem, dims)


def const_node(name, np_type, values, shape=None):
    import numpy as np
    arr = np.array(values, dtype=np_type)
    if shape is not None:
        arr = arr.reshape(shape)
    return helper.make_node(
        'Constant', [], [name], name=f'Const_{name}', value=onnx.numpy_helper.from_array(arr, name=name)
    )


def build_body_graph(dhqk: int, dhv: int, nh: int, eps: float, opset: int):
    # State inputs
    c_in = make_vi('c_in', TensorProto.FLOAT, ['B', nh, dhqk, dhv])
    n_in = make_vi('n_in', TensorProto.FLOAT, ['B', nh, dhqk])
    m_in = make_vi('m_in', TensorProto.FLOAT, ['B', nh, 1])
    # Scan inputs (time slice)
    q_t = make_vi('q_t', TensorProto.FLOAT, ['B', nh, dhqk])
    k_t = make_vi('k_t', TensorProto.FLOAT, ['B', nh, dhqk])
    v_t = make_vi('v_t', TensorProto.FLOAT, ['B', nh, dhv])
    i_t = make_vi('i_t', TensorProto.FLOAT, ['B', nh, 1])
    f_t = make_vi('f_t', TensorProto.FLOAT, ['B', nh, 1])

    # Outputs: new states then sequence output per step
    c_out = make_vi('c_out', TensorProto.FLOAT, ['B', nh, dhqk, dhv])
    n_out = make_vi('n_out', TensorProto.FLOAT, ['B', nh, dhqk])
    m_out = make_vi('m_out', TensorProto.FLOAT, ['B', nh, 1])
    h_t = make_vi('h_t', TensorProto.FLOAT, ['B', nh, dhv])

    nodes = []
    # constants
    nodes.append(const_node('one', 'float32', [1.0]))
    nodes.append(const_node('eps', 'float32', [eps]))
    nodes.append(const_node('inv_sqrt_d', 'float32', [1.0 / math.sqrt(float(dhqk))]))

    # f_logsig = -log(1 + exp(-f_t))
    f_neg = helper.make_node('Neg', ['f_t'], ['f_neg'], name='Neg_f')
    f_exp = helper.make_node('Exp', ['f_neg'], ['f_exp'], name='Exp_fneg')
    f_one = helper.make_node('Add', ['f_exp', 'one'], ['f_one'], name='Add_one')
    f_log = helper.make_node('Log', ['f_one'], ['f_log'], name='Log')
    f_logsig = helper.make_node('Neg', ['f_log'], ['f_logsig'], name='Neg_log')
    nodes += [f_neg, f_exp, f_one, f_log, f_logsig]

    # m_new = max(f_logsig + m_in, i_t)
    fpm = helper.make_node('Add', ['f_logsig', 'm_in'], ['fpm'], name='Add_fm')
    m_new = helper.make_node('Max', ['fpm', 'i_t'], ['m_new'], name='Max_m')
    nodes += [fpm, m_new]

    # F_act = exp(f_logsig + m_in - m_new); I_act = exp(i_t - m_new)
    fpm2 = helper.make_node('Add', ['f_logsig', 'm_in'], ['fpm2'], name='Add_fm2')
    fm_minus_m = helper.make_node('Sub', ['fpm2', 'm_new'], ['fm_minus_m'], name='Sub_fm_m')
    F_act = helper.make_node('Exp', ['fm_minus_m'], ['F_act'], name='Exp_F')
    i_minus_m = helper.make_node('Sub', ['i_t', 'm_new'], ['i_minus_m'], name='Sub_i_m')
    I_act = helper.make_node('Exp', ['i_minus_m'], ['I_act'], name='Exp_I')
    nodes += [fpm2, fm_minus_m, F_act, i_minus_m, I_act]

    # q_s = q_t * inv_sqrt_d
    q_s = helper.make_node('Mul', ['q_t', 'inv_sqrt_d'], ['q_s'], name='Mul_qs')
    nodes.append(q_s)

    # kv outer product -> (B,NH,DHQK,DHV)
    k_e = helper.make_node('Unsqueeze', ['k_t'], ['k_e'], name='Unsq_k', axes=[-1])
    v_e1 = helper.make_node('Unsqueeze', ['v_t'], ['v_e1'], name='Unsq_v1', axes=[-2])
    v_e = helper.make_node('Unsqueeze', ['v_e1'], ['v_e'], name='Unsq_v2', axes=[])
    kv = helper.make_node('MatMul', ['k_e', 'v_e'], ['kv'], name='MatMul_kv')
    nodes += [k_e, v_e1, v_e, kv]

    # Broadcast gates for c_new
    F_e1 = helper.make_node('Unsqueeze', ['F_act'], ['F_e1'], name='Unsq_F1', axes=[-1])
    F_b = helper.make_node('Unsqueeze', ['F_e1'], ['F_b'], name='Unsq_F2', axes=[-1])
    I_e1 = helper.make_node('Unsqueeze', ['I_act'], ['I_e1'], name='Unsq_I1', axes=[-1])
    I_b = helper.make_node('Unsqueeze', ['I_e1'], ['I_b'], name='Unsq_I2', axes=[-1])
    nodes += [F_e1, F_b, I_e1, I_b]

    F_c = helper.make_node('Mul', ['F_b', 'c_in'], ['F_c'], name='Mul_Fc')
    I_kv = helper.make_node('Mul', ['I_b', 'kv'], ['I_kv'], name='Mul_Ikv')
    c_new_node = helper.make_node('Add', ['F_c', 'I_kv'], ['c_out'], name='Add_cnew')
    nodes += [F_c, I_kv, c_new_node]

    # n_new
    Fn = helper.make_node('Mul', ['F_act', 'n_in'], ['Fn'], name='Mul_Fn')
    Ik = helper.make_node('Mul', ['I_act', 'k_t'], ['Ik'], name='Mul_Ik')
    n_new_node = helper.make_node('Add', ['Fn', 'Ik'], ['n_out'], name='Add_nnew')
    nodes += [Fn, Ik, n_new_node]

    # h_t
    q_s_e = helper.make_node('Unsqueeze', ['q_s'], ['q_s_e'], name='Unsq_qs', axes=[-2])
    h_num_e = helper.make_node('MatMul', ['q_s_e', 'c_out'], ['h_num_e'], name='MatMul_qc')
    h_num = helper.make_node('Squeeze', ['h_num_e'], ['h_num'], name='Squeeze_hnum', axes=[-2])
    n_new_e = helper.make_node('Unsqueeze', ['n_out'], ['n_new_e'], name='Unsq_n', axes=[-1])
    qn_e = helper.make_node('MatMul', ['q_s_e', 'n_new_e'], ['qn_e'], name='MatMul_qn')
    qn = helper.make_node('Squeeze', ['qn_e'], ['qn'], name='Squeeze_qn', axes=[-1, -2])
    m_neg = helper.make_node('Neg', ['m_new'], ['m_neg'], name='Neg_m')
    max_val = helper.make_node('Exp', ['m_neg'], ['max_val'], name='Exp_negm')
    qn_abs = helper.make_node('Abs', ['qn'], ['qn_abs'], name='Abs_qn')
    denom0 = helper.make_node('Max', ['qn_abs', 'max_val'], ['denom0'], name='Max_den')
    h_denom = helper.make_node('Add', ['denom0', 'eps'], ['h_denom'], name='Add_eps')
    h_t_node = helper.make_node('Div', ['h_num', 'h_denom'], ['h_t'], name='Div_h')
    nodes += [q_s_e, h_num_e, h_num, n_new_e, qn_e, qn, m_neg, max_val, qn_abs, denom0, h_denom, h_t_node]

    body = helper.make_graph(
        nodes,
        'mLSTM_scan_body',
        inputs=[c_in, n_in, m_in, q_t, k_t, v_t, i_t, f_t],
        outputs=[c_out, n_out, m_out, h_t],
    )
    return body


def build_model(dhqk: int, dhv: int, nh: int, eps: float = 1e-6, opset: int = 13) -> onnx.ModelProto:
    S = 'S'
    B = 'B'
    Q = make_vi('Q', TensorProto.FLOAT, [S, B, nh, dhqk])
    K = make_vi('K', TensorProto.FLOAT, [S, B, nh, dhqk])
    V = make_vi('V', TensorProto.FLOAT, [S, B, nh, dhv])
    I = make_vi('I', TensorProto.FLOAT, [S, B, nh, 1])
    F = make_vi('F', TensorProto.FLOAT, [S, B, nh, 1])
    C_init = make_vi('C_init', TensorProto.FLOAT, [B, nh, dhqk, dhv])
    N_init = make_vi('N_init', TensorProto.FLOAT, [B, nh, dhqk])
    M_init = make_vi('M_init', TensorProto.FLOAT, [B, nh, 1])

    H = make_vi('H', TensorProto.FLOAT, [S, B, nh, dhv])
    C_out = make_vi('C_out', TensorProto.FLOAT, [B, nh, dhqk, dhv])
    N_out = make_vi('N_out', TensorProto.FLOAT, [B, nh, dhqk])
    M_out = make_vi('M_out', TensorProto.FLOAT, [B, nh, 1])

    body = build_body_graph(dhqk, dhv, nh, eps, opset)
    scan_node = helper.make_node(
        'Scan',
        inputs=['C_init', 'N_init', 'M_init', 'Q', 'K', 'V', 'I', 'F'],
        outputs=['C_out', 'N_out', 'M_out', 'H'],
        num_scan_inputs=5,
        body=body,
        name='Scan_mLSTM',
    )

    graph = helper.make_graph(
        [scan_node],
        'mLSTM_scan',
        inputs=[Q, K, V, I, F, C_init, N_init, M_init],
        outputs=[H, C_out, N_out, M_out],
    )

    model = helper.make_model(graph, producer_name='xlstm-onnx', opset_imports=[helper.make_opsetid('', opset)])
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dhqk', type=int, default=128)
    ap.add_argument('--dhv', type=int, default=128)
    ap.add_argument('--nh', type=int, default=4)
    ap.add_argument('--out', type=str, default='mlstm_scan.onnx')
    args = ap.parse_args()
    model = build_model(dhqk=args.dhqk, dhv=args.dhv, nh=args.nh)
    onnx.save(model, args.out)
    print(f'wrote {args.out}')

