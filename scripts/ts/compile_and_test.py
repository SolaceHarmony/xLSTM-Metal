from __future__ import annotations

"""
Compile xLSTMTorch to TorchScript and run a quick sanity on CPU/MPS.

Usage:
  PYTHONPATH=. python scripts/ts/compile_and_test.py --device cpu
"""

import argparse
import time
import torch

from xlstm_solace_torch.models.model import xLSTMTorch, xLSTMTorchConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="intra-op threads")
    ap.add_argument("--interop", type=int, default=1, help="inter-op threads")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--V", type=int, default=256)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--L", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)

    cfg = xLSTMTorchConfig(
        embedding_dim=args.D,
        num_heads=args.H,
        num_blocks=args.L,
        vocab_size=args.V,
        # TS-friendly defaults (native paths)
        chunkwise_kernel="chunkwise--native_compiled_autograd",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        mode="inference",
        return_last_states=True,
    )
    model = xLSTMTorch(cfg).to(args.device).eval()

    # Script
    scripted = torch.jit.script(model)
    scripted = torch.jit.optimize_for_inference(scripted)

    # Sanity
    x = torch.randint(0, args.V, (args.B, args.S), dtype=torch.long, device=args.device)
    with torch.no_grad():
        t0 = time.time()
        logits, _ = scripted.forward_with_state(x, [None for _ in range(args.L)])
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        dt = time.time() - t0
    print(f"ok: logits={tuple(logits.shape)} time={dt:.4f}s threads={args.threads} interop={args.interop}")


if __name__ == "__main__":
    main()

