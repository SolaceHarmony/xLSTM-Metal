"""
Run mLSTM step ONNX with ONNX Runtime using the Core ML execution provider.

Usage:
  PYTHONPATH=. python onnx/run_step_ort_coreml.py --model mlstm_step.onnx --B 1 --NH 4 --DHQK 128 --DHV 128

Requires onnxruntime with CoreML EP available.
"""

import argparse
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--NH', type=int, default=4)
    ap.add_argument('--DHQK', type=int, default=128)
    ap.add_argument('--DHV', type=int, default=128)
    args = ap.parse_args()

    sess_options = ort.SessionOptions()
    providers = ['CoreMLExecutionProvider']
    sess = ort.InferenceSession(args.model, sess_options, providers=providers)
    assert 'CoreMLExecutionProvider' in sess.get_providers(), 'CoreML EP not available; ensure ORT is built with CoreML'

    B, NH, DHQK, DHV = args.B, args.NH, args.DHQK, args.DHV
    q = np.random.randn(B, NH, DHQK).astype('float32')
    k = np.random.randn(B, NH, DHQK).astype('float32')
    v = np.random.randn(B, NH, DHV).astype('float32')
    i = np.random.randn(B, NH, 1).astype('float32')
    f = np.random.randn(B, NH, 1).astype('float32')
    c_old = np.random.randn(B, NH, DHQK, DHV).astype('float32')
    n_old = np.random.randn(B, NH, DHQK).astype('float32')
    m_old = np.random.randn(B, NH, 1).astype('float32')

    outs = sess.run(None, {
        'q': q, 'k': k, 'v': v, 'i': i, 'f': f,
        'c_old': c_old, 'n_old': n_old, 'm_old': m_old,
    })
    h, c_new, n_new, m_new = outs
    print('h:', h.shape, 'c_new:', c_new.shape, 'n_new:', n_new.shape, 'm_new:', m_new.shape)


if __name__ == '__main__':
    main()

