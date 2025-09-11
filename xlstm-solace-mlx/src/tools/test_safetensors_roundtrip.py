"""
Round-trip test for MLX .safetensors weight loading on the Solace xLSTM model.

Creates a tiny model, saves weights, mutates a parameter, loads weights back
with strict=True, and verifies parameters are restored.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

from xlstm_solace_mlx.api import create_xlstm_model


def param_checksum(params: list[mx.array]) -> float:
    total = mx.array(0.0, dtype=mx.float32)
    for p in params:
        total = total + mx.sum(p.astype(mx.float32))
    return float(total)


def main() -> None:
    # Tiny model for a quick test
    model = create_xlstm_model(
        vocab_size=128,
        num_layers=2,
        signature=(1, 1),
        inp_dim=64,
        head_dim=16,
        head_num=4,
        dropout=0.0,
    )

    # Collect parameters
    params_before = [p for _, p in model.parameters()]
    sum_before = param_checksum(params_before)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    wt_path = out_dir / "test_xlstm_mlx.safetensors"

    # Save current weights
    model.save_weights(str(wt_path))

    # Mutate a parameter deterministically
    name0, p0 = next(iter(model.parameters()))
    p0 += mx.array(1e-2, dtype=p0.dtype)

    params_mid = [p for _, p in model.parameters()]
    sum_mid = param_checksum(params_mid)
    assert abs(sum_mid - sum_before) > 1e-6, "Parameter mutation did not change checksum"

    # Reload saved weights strictly
    model.load_weights(str(wt_path), strict=True)

    params_after = [p for _, p in model.parameters()]
    sum_after = param_checksum(params_after)

    # Checksum should be restored exactly
    assert abs(sum_after - sum_before) < 1e-6, f"Reload failed: {sum_after} vs {sum_before}"
    print("OK: .safetensors round-trip passed.")


if __name__ == "__main__":
    main()

