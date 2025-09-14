"""
Run mLSTM sequence ONNX (Scan) with ONNX Runtime Core ML EP.

Usage:
  PYTHONPATH=. python onnx/run_scan_ort_coreml.py --model mlstm_scan.onnx --S 64 --B 1 --NH 4 --DHQK 128 --DHV 128
"""

import argparse
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--S', type=int, default=64)
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--NH', type=int, default=4)
    ap.add_argument('--DHQK', type=int, default=128)
    ap.add_argument('--DHV', type=int, default=128)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=['CoreMLExecutionProvider'])
    assert 'CoreMLExecutionProvider' in sess.get_providers(), 'CoreML EP not available'

    S, B, NH, DHQK, DHV = args.S, args.B, args.NH, args.DHQK, args.DHV
    Q = np.random.randn(S, B, NH, DHQK).astype('float32')
    K = np.random.randn(S, B, NH, DHQK).astype('float32')
    V = np.random.randn(S, B, NH, DHV).astype('float32')
    I = np.random.randn(S, B, NH, 1).astype('float32')
    F = np.random.randn(S, B, NH, 1).astype('float32')
    C_init = np.random.randn(B, NH, DHQK, DHV).astype('float32')
    N_init = np.random.randn(B, NH, DHQK).astype('float32')
    M_init = np.random.randn(B, NH, 1).astype('float32')

    H, C_out, N_out, M_out = sess.run(None, {
        'Q': Q, 'K': K, 'V': V, 'I': I, 'F': F,
        'C_init': C_init, 'N_init': N_init, 'M_init': M_init,
    })
    print('H:', H.shape, 'C_out:', C_out.shape, 'N_out:', N_out.shape, 'M_out:', M_out.shape)


if __name__ == '__main__':
    main()

