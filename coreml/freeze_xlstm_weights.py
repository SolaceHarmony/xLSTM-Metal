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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--weights', type=str, help='path to weights .npz')
    g.add_argument('--safetensors', nargs='+', help='one or more .safetensors shards')
    ap.add_argument('--V', type=int, required=True)
    ap.add_argument('--D', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--NH', type=int, required=True)
    ap.add_argument('--DHQK', type=int, required=True)
    ap.add_argument('--DHV', type=int, required=True)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--block-types', type=str, default=None, help='comma-separated 0/1 list for L blocks')
    ap.add_argument('--profile', type=str, default=None, help='xlstm profile JSON with slstm_at/num_blocks to derive schedule')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    # Derive schedule
    if args.block_types is not None:
        block_types = [int(x) for x in args.block_types.split(',') if x.strip()!='']
        assert len(block_types) == args.L
    elif args.profile is not None:
        import json
        prof = json.loads(open(args.profile, 'r').read())
        assert prof.get('num_blocks', args.L) == args.L
        sl = prof.get('slstm_at', [])
        sl_list = list(range(args.L)) if sl == 'all' else [int(i) for i in sl]
        block_types = [1 if i in sl_list else 0 for i in range(args.L)]
    else:
        raise SystemExit('Provide either --block-types or --profile to determine schedule')

    # Load weights
    weights: Dict[str, np.ndarray]
    if args.weights:
        weights = load_weights_npz(args.weights)
    else:
        try:
            from safetensors.numpy import load_file
        except Exception as e:
            raise SystemExit('Please install safetensors to load --safetensors shards')
        weights = {}
        for shard in args.safetensors:
            shard_tensors = load_file(shard)
            weights.update({k: np.array(v) for k, v in shard_tensors.items()})

    # TODO: Map `weights` dict into MIL constants: embed_W, lm_W, per-block LN, projections,
    # FFN weights, conv kernels/bias (transpose/reshape to (K,D) as needed), and rebuild
    # the program with mb.const for each parameter instead of TensorSpecs.

    # For now, build program with TensorSpecs (weights external); next pass will freeze.
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
