xlstm-solace-mlx — import xlstm_solace_mlx

Purpose
- PyPI-ready packaging for the MLX path (Apple MLX, no Ray), import name `xlstm_solace_mlx`.

Disclaimer
- Independent Solace fork; not affiliated with NX‑AI or the original xLSTM authors.
- Name intentionally includes “solace” to avoid confusion with upstream.

Status
- Pre-release scaffolding. CLI wiring and module exports will be added after the src/ refactor.

Notes
- Target Python: 3.12 (conda base recommended).
- Focus: pure-MLX, single-process execution; keep MLX tensors on device (avoid CPU/NumPy hops).
