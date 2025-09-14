from __future__ import annotations

"""
Thread sweep for upstream TorchScript typed-state forward/generate.

Usage:
  PYTHONPATH=. python scripts/bench/ts_thread_sweep.py --device cpu
"""

import argparse
import time
import itertools
import torch

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


def bench(device: str, B: int, S: int, V: int, D: int, H: int, L: int, threads_list, interop_list):
    cfg = xLSTMLMModelConfig(
        embedding_dim=D,
        num_blocks=L,
        num_heads=H,
        vocab_size=V,
        slstm_at=[],
    )
    model = xLSTMLMModel(cfg).to(device).eval()
    scripted = torch.jit.script(model)
    scripted = torch.jit.optimize_for_inference(scripted)

    x = torch.randint(0, V, (B, S), dtype=torch.long, device=device)
    nblocks = len(model.xlstm_block_stack.blocks)

    for t, inter in itertools.product(threads_list, interop_list):
        torch.set_num_threads(t)
        torch.set_num_interop_threads(inter)
        mlstm_states = [None for _ in range(nblocks)]
        conv_states = [None for _ in range(nblocks)]
        slstm_states = [None for _ in range(nblocks)]
        with torch.no_grad():
            t0 = time.time()
            logits, mlstm_states, conv_states, slstm_states = scripted.forward_with_states(x, mlstm_states, conv_states, slstm_states)
            dt = time.time() - t0
        print(f"threads={t:2d} interop={inter:2d} fwd_ms={dt*1000:.2f} logits={tuple(logits.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--S', type=int, default=128)
    ap.add_argument('--V', type=int, default=50304)
    ap.add_argument('--D', type=int, default=4096)
    ap.add_argument('--H', type=int, default=8)
    ap.add_argument('--L', type=int, default=32)
    args = ap.parse_args()
    threads_list = [1, 2, 4, 8]
    interop_list = [1, 2, 4]
    bench(args.device, args.B, args.S, args.V, args.D, args.H, args.L, threads_list, interop_list)


if __name__ == '__main__':
    main()

