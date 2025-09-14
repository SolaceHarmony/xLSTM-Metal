"""
Export the MIL xLSTM decode step (mLSTM + sLSTM capable) to a stateful MLProgram.

This declares per-block recurrent states as Core ML states (ct.StateType), so the
runtime keeps them persistent across calls and updates them in-place.

Usage:
  PYTHONPATH=. python coreml/export_xlstm_decode_step.py \
    --V 50304 --D 4096 --L 32 --NH 8 --DHQK 512 --DHV 512 \
    --out xlstm_decode_step.mlpackage
"""

import argparse
import coremltools as ct
import numpy as np

from coreml.build_xlstm_decode_step import build_xlstm_decode_step_program


def export_decode_step(V: int, D: int, L: int, NH: int, DHQK: int, DHV: int, out_path: str,
                       min_ios: str = "18.0"):
    prog, _ = build_xlstm_decode_step_program(V=V, D=D, L=L, NH=NH, DHQK=DHQK, DHV=DHV)

    B = ct.RangeDim(lower_bound=1, upper_bound=ct.RangeDim.MAX)
    F32 = np.float32
    I32 = np.int32

    # Minimal input dtypes/shapes; weight placeholders are inferred. If you prefer strict
    # typing, add ct.TensorType entries for all weights as well.
    inputs = [
        ct.TensorType(name="tok_id", shape=(B, 1), dtype=I32),
        ct.TensorType(name="embed_W", shape=(V, D), dtype=F32),
        ct.TensorType(name="lm_W", shape=(D, V), dtype=F32),
    ]

    # Declare persistent states: per-block c_i, n_i, m_i
    states = []
    for i in range(L):
        states.append(ct.StateType(name=f"c_{i}", shape=(B, NH, DHQK, DHV), dtype=F32))
        states.append(ct.StateType(name=f"n_{i}", shape=(B, NH, DHQK), dtype=F32))
        states.append(ct.StateType(name=f"m_{i}", shape=(B, NH, 1), dtype=F32))

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=inputs,
        states=states,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOSVersion(min_ios),
    )

    if out_path.endswith(".mlpackage") or out_path.endswith(".mlmodel"):
        model.save(out_path)
    else:
        model.save(out_path + ".mlpackage")
    print(f"Saved MLProgram decode step to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=256)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--NH", type=int, default=4)
    ap.add_argument("--DHQK", type=int, default=32)
    ap.add_argument("--DHV", type=int, default=32)
    ap.add_argument("--out", type=str, default="xlstm_decode_step.mlpackage")
    ap.add_argument("--min-ios", type=str, default="18.0")
    args = ap.parse_args()
    export_decode_step(V=args.V, D=args.D, L=args.L, NH=args.NH, DHQK=args.DHQK, DHV=args.DHV, out_path=args.out, min_ios=args.min_ios)

