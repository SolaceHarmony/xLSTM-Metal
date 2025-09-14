"""
Freeze xlstm weights into MIL constants for a self-contained stateful MLProgram.

This script loads trained weights (exported to .npz or .npy bundles) and rebuilds
the stateful MIL decode graph with mb.const for all parameters (embeddings, per-
block layernorm gammas/betas, projections, FFN weights, conv kernels), then
converts to an MLProgram that contains weights internally.

Note: This is a scaffold. Wire your own loader that maps xlstm parameter names
to the placeholders expected by build_stateful_xlstm_decode_program.
"""

from typing import List, Dict
import argparse
import numpy as np
import coremltools as ct

from coreml.build_xlstm_decode_stateful import build_stateful_xlstm_decode_program


def load_weights_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='path to weights .npz')
    ap.add_argument('--V', type=int, required=True)
    ap.add_argument('--D', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--NH', type=int, required=True)
    ap.add_argument('--DHQK', type=int, required=True)
    ap.add_argument('--DHV', type=int, required=True)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--block-types', type=str, required=True, help='comma-separated 0/1 list for L blocks')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    block_types = [int(x) for x in args.block_types.split(',') if x.strip()!='']
    assert len(block_types) == args.L

    # Build program with TensorSpecs
    prog = build_stateful_xlstm_decode_program(V=args.V, D=args.D, L=args.L, NH=args.NH,
                                               DHQK=args.DHQK, DHV=args.DHV, K=args.K,
                                               block_types=block_types)

    # Convert with default inputs (weights as external tensors)
    # In a full freeze pass, rebuild the program with mb.const for each param
    # using the loaded numpy arrays mapped to the expected names.
    mlmodel = ct.convert(prog, convert_to='mlprogram', compute_units=ct.ComputeUnit.ALL,
                         minimum_deployment_target=ct.target.iOS18)
    mlmodel.save(args.out if args.out.endswith(('.mlmodel','.mlpackage')) else args.out + '.mlpackage')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()

