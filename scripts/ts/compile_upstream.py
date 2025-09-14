from __future__ import annotations

"""
Script upstream xLSTMLMModel and run a typed-state forward and greedy step.

Usage:
  PYTHONPATH=. python scripts/ts/compile_upstream.py --device cpu
"""

import argparse
import time
import torch

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--interop", type=int, default=1)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--V", type=int, default=256)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--decode", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)

    cfg = xLSTMLMModelConfig(
        embedding_dim=args.D,
        num_blocks=args.L,
        num_heads=args.H,
        vocab_size=args.V,
        add_post_blocks_norm=True,
        slstm_at=[],  # mLSTM-only for TS sanity
    )
    model = xLSTMLMModel(cfg).to(args.device).eval()

    scripted = torch.jit.script(model)
    scripted = torch.jit.optimize_for_inference(scripted)

    x = torch.randint(0, args.V, (args.B, args.S), dtype=torch.long, device=args.device)
    nblocks = len(model.xlstm_block_stack.blocks)
    mlstm_states = [None for _ in range(nblocks)]
    conv_states = [None for _ in range(nblocks)]
    slstm_states = [None for _ in range(nblocks)]

    with torch.no_grad():
        t0 = time.time()
        logits, mlstm_states, conv_states, slstm_states = scripted.forward_with_states(
            x, mlstm_states, conv_states, slstm_states
        )
        dt = time.time() - t0
    print(f"forward_with_states ok: logits={tuple(logits.shape)} time={dt:.4f}s threads={args.threads} interop={args.interop}")

    with torch.no_grad():
        t0 = time.time()
        gen, mlstm_states, conv_states, slstm_states = scripted.generate_greedy(x[:, :1], args.decode)
        dt = time.time() - t0
    print(f"generate_greedy ok: tokens={tuple(gen.shape)} time={dt:.4f}s")


if __name__ == "__main__":
    main()

