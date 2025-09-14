"""
Export the MIL mLSTM step to a Core ML MLProgram (.mlpackage / .mlmodel).

Builds the MIL Program from build_mlstm_step.py and converts it with
coremltools to an MLProgram with flexible batch dimension and fixed NH/DHQK/DHV.

Usage:
  PYTHONPATH=. python coreml/export_mlstm_step.py \
    --dhqk 512 --dhv 512 --nh 8 --eps 1e-6 --out mlstm_step.mlpackage

Notes:
  - This requires coremltools >= 6.x and macOS with Xcode toolchain.
  - Set compute_units to ALL to allow ANE/GPU/CPU selection at runtime.
"""

import argparse
import numpy as np

import coremltools as ct
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

from coreml.build_mlstm_step import build_mlstm_step_program


def export_mlstm_step(dhqk: int, dhv: int, nh: int, eps: float, out_path: str, minimum_deployment_target: str = "13.0"):
    # Build MIL Program
    prog, _ = build_mlstm_step_program(dhqk=dhqk, dhv=dhv, nh=nh, eps=eps)

    B = ct.RangeDim(lower_bound=1, upper_bound=ct.RangeDim.MAX)
    F32 = np.float32

    inputs = [
        ct.TensorType(name="q", shape=(B, nh, dhqk), dtype=F32),
        ct.TensorType(name="k", shape=(B, nh, dhqk), dtype=F32),
        ct.TensorType(name="v", shape=(B, nh, dhv), dtype=F32),
        ct.TensorType(name="i", shape=(B, nh, 1), dtype=F32),
        ct.TensorType(name="f", shape=(B, nh, 1), dtype=F32),
        ct.TensorType(name="c_old", shape=(B, nh, dhqk, dhv), dtype=F32),
        ct.TensorType(name="n_old", shape=(B, nh, dhqk), dtype=F32),
        ct.TensorType(name="m_old", shape=(B, nh, 1), dtype=F32),
    ]

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=inputs,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOSVersion(minimum_deployment_target),
    )

    # Save as mlpackage (directory) or mlmodel (single file)
    if out_path.endswith(".mlpackage") or out_path.endswith(".mlmodel"):
        model.save(out_path)
    else:
        model.save(out_path + ".mlpackage")

    print(f"Saved MLProgram to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dhqk", type=int, default=128)
    ap.add_argument("--dhv", type=int, default=128)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--out", type=str, default="mlstm_step.mlpackage")
    ap.add_argument("--min-ios", type=str, default="13.0")
    args = ap.parse_args()
    export_mlstm_step(dhqk=args.dhqk, dhv=args.dhv, nh=args.nh, eps=args.eps, out_path=args.out, minimum_deployment_target=args.min_ios)

