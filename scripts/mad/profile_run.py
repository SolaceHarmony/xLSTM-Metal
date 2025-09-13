from __future__ import annotations

"""
Run a simple capability sanity check for an xLSTM profile on synthetic tasks.

Usage:
  PYTHONPATH=. python scripts/mad/profile_run.py --profile path/to/profile.json --task in_context_recall

Policy:
- No shims; fail fast on bad inputs.
- Minimal dependencies; uses in-repo task generators.
"""

import argparse
import json
from pathlib import Path

import torch

from xlstm.profile_loader import build_lm_config
from xlstm.xlstm_lm_model import xLSTMLMModel
from scripts.mad.tasks import TaskConfig, generate_in_context_recall, accuracy_ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=str)
    ap.add_argument("--task", choices=["in_context_recall"], default="in_context_recall")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32)
    args = ap.parse_args()

    prof_path = Path(args.profile)
    prof = json.loads(prof_path.read_text())

    # Override basic dims from CLI if provided (explicit beats implicit)
    prof["embedding_dim"] = int(prof.get("embedding_dim", 128))
    prof["num_blocks"] = int(prof.get("num_blocks", 2))
    prof["context_length"] = int(prof.get("context_length", args.seq))
    prof["vocab_size"] = int(prof.get("vocab_size", args.vocab))

    lm_cfg = build_lm_config(prof)
    model = xLSTMLMModel(lm_cfg)
    model.to(args.device)
    model.eval()

    # Task generation (eval mode)
    tcfg = TaskConfig(vocab_size=lm_cfg.vocab_size, seq_len=args.seq, batch_size=args.batch)
    x, y = generate_in_context_recall(tcfg, train=False)
    x = x.to(args.device)
    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(dim=-1)

    acc = accuracy_ignore(pred.cpu(), y, ignore_index=-100)
    print(f"task={args.task} batch={args.batch} seq={args.seq} vocab={tcfg.vocab_size} acc={acc:.4f}")


if __name__ == "__main__":
    main()

