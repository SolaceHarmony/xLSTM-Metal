"""
Export the stateful MIL xLSTM decode step (LayerNorm + mLSTM/sLSTM + FFN) to MLProgram.

This expects the block schedule as a comma-separated list (e.g., "0,0,1,0,..."),
where 0 = mLSTM, 1 = sLSTM. All recurrent states are native MIL states and will
be updated in-place by Core ML (no ct.StateType needed).

Usage:
  PYTHONPATH=. python coreml/export_xlstm_decode_stateful.py \
    --V 50304 --D 4096 --L 32 --NH 8 --DHQK 512 --DHV 512 --K 4 \
    --block-types "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" \
    --out xlstm_decode_stateful.mlpackage
"""

import argparse
import coremltools as ct

from coreml.build_xlstm_decode_stateful import build_stateful_xlstm_decode_program
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--V', type=int, required=True)
    ap.add_argument('--D', type=int, required=True)
    ap.add_argument('--L', type=int, required=True)
    ap.add_argument('--NH', type=int, required=True)
    ap.add_argument('--DHQK', type=int, required=True)
    ap.add_argument('--DHV', type=int, required=True)
    ap.add_argument('--K', type=int, default=4, help='causal conv kernel size')
    ap.add_argument('--block-types', type=str, default=None, help='comma-separated 0/1 list (0=mLSTM, 1=sLSTM)')
    ap.add_argument('--profile', type=str, default=None, help='xlstm profile JSON with slstm_at/num_blocks to derive schedule')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--min-ios', type=str, default='18.0')
    args = ap.parse_args()

    if args.block_types is not None:
        block_types = [int(x) for x in args.block_types.split(',') if x.strip()!='']
        assert len(block_types) == args.L, 'block-types length must equal L'
    elif args.profile is not None:
        prof = json.loads(open(args.profile, 'r').read())
        assert 'num_blocks' in prof, 'profile must contain num_blocks'
        assert prof['num_blocks'] == args.L, 'L must match profile num_blocks'
        sl = prof.get('slstm_at', [])
        if sl == 'all':
            sl_list = list(range(args.L))
        else:
            sl_list = [int(i) for i in sl]
        block_types = [1 if i in sl_list else 0 for i in range(args.L)]
    else:
        raise SystemExit('Provide either --block-types or --profile to determine schedule')

    prog = build_stateful_xlstm_decode_program(
        V=args.V, D=args.D, L=args.L, NH=args.NH, DHQK=args.DHQK, DHV=args.DHV,
        K=args.K, block_types=block_types,
    )

    mlmodel = ct.convert(
        prog,
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOSVersion(args.min_ios),
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(args.out if args.out.endswith(('.mlmodel', '.mlpackage')) else args.out + '.mlpackage')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
