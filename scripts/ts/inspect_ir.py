from __future__ import annotations

"""
Inspect TorchScript IR for upstream LM typed-state path.

Usage:
  PYTHONPATH=. python scripts/ts/inspect_ir.py
"""

import torch
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


def main():
    cfg = xLSTMLMModelConfig(embedding_dim=128, num_heads=4, num_blocks=2, vocab_size=256, slstm_at=[])
    model = xLSTMLMModel(cfg).eval()
    scripted = torch.jit.script(model)
    print('forward_with_states graph:')
    print(scripted.forward_with_states.graph)


if __name__ == '__main__':
    main()

