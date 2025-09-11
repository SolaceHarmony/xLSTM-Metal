"""
Simple runner for the Solace Torch (MPS + Ray) entry.

Usage:
  python xlstm_run_pt.py --model_path ./xlstm_7b_model --prompt "..." --max_new_tokens 32
"""
from __future__ import annotations

import sys

import os
root = os.path.dirname(os.path.abspath(__file__))
torch_src = os.path.join(root, "xlstm-solace-torch", "src")
if torch_src not in sys.path:
    sys.path.insert(0, torch_src)

from xlstm_generate_pt import main


if __name__ == "__main__":
    sys.exit(main())
