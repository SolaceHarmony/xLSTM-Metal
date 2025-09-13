"""
Simple runner for the Solace MLX entry.

Usage:
  python xlstm_run_mlx.py --prompt "..." --max_new_tokens 64
"""
from __future__ import annotations

import sys

import os
root = os.path.dirname(os.path.abspath(__file__))
mlx_src = os.path.join(root, "xlstm-solace-mlx", "src")
if mlx_src not in sys.path:
    sys.path.insert(0, mlx_src)

from xlstm_mlx.cli import main


if __name__ == "__main__":
    sys.exit(main())
