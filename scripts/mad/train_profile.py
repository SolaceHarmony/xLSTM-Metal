from __future__ import annotations

"""
Minimal training harness to score an xLSTM profile on MAD-style tasks quickly.

Policy:
- No shims; fail fast.
- Small, fixed training loop (steps), synthetic batches on the fly.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from xlstm.profile_loader import build_lm_config
from xlstm.xlstm_lm_model import xLSTMLMModel
from scripts.mad.tasks import (
    TaskConfig,
    generate_in_context_recall,
    generate_selective_copying,
    accuracy_ignore,
    IGNORE_INDEX,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=str)
    ap.add_argument("--task", choices=["in_context_recall", "selective_copying"], default="in_context_recall")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    args = ap.parse_args()

    prof_path = Path(args.profile)
    prof = json.loads(prof_path.read_text())
    prof["embedding_dim"] = int(prof.get("embedding_dim", 128))
    prof["num_blocks"] = int(prof.get("num_blocks", 2))
    prof["context_length"] = int(prof.get("context_length", args.seq))
    prof["vocab_size"] = int(prof.get("vocab_size", args.vocab))

    lm_cfg = build_lm_config(prof)
    model = xLSTMLMModel(lm_cfg)
    device = torch.device(args.device)
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    tcfg = TaskConfig(vocab_size=lm_cfg.vocab_size, seq_len=args.seq, batch_size=args.batch)

    for step in range(1, args.steps + 1):
        if args.task == "in_context_recall":
            x, y = generate_in_context_recall(tcfg, train=True)
            # typical LM shift: x is length S-1, targets y is length S-1
        elif args.task == "selective_copying":
            x, y = generate_selective_copying(tcfg)
        else:
            raise ValueError("unsupported task")

        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()

        if step % max(1, args.steps // 10) == 0:
            with torch.no_grad():
                model.eval()
                if args.task == "in_context_recall":
                    x_eval, y_eval = generate_in_context_recall(tcfg, train=False)
                else:
                    x_eval, y_eval = generate_selective_copying(tcfg)
                x_eval = x_eval.to(device)
                logits_eval = model(x_eval)
                pred = logits_eval.argmax(dim=-1).cpu()
                acc = accuracy_ignore(pred, y_eval, ignore_index=IGNORE_INDEX)
                print(f"step={step}/{args.steps} loss={loss.item():.4f} acc={acc:.4f}")
                model.train()


if __name__ == "__main__":
    main()

