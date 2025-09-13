xlstm-solace-mlx — import xlstm_mlx

Purpose
- PyPI-ready packaging for the MLX path (Apple MLX, no Ray), import name `xlstm_mlx`.

Disclaimer
- Independent Solace fork; not affiliated with NX‑AI or the original xLSTM authors.
- Name intentionally includes “solace” to avoid confusion with upstream.

Status
- Pre-release scaffolding. CLI wiring and module exports will be added after the src/ refactor.

Notes
- Target Python: 3.12 (conda base recommended).
- Focus: pure-MLX, single-process execution; keep MLX tensors on device (avoid CPU/NumPy hops).

Weights (.safetensors)
- The CLI can load MLX-native .safetensors weights produced for this architecture.
- Usage: `python xlstm_run_mlx.py --weights /path/to/model.safetensors --strict 1 --prompt "Hello"`.
- Strict loading (`--strict 1`, default) checks names/shapes match the module’s parameters.
- If converting from Hugging Face/PyTorch, use `mlx_lm.convert` to produce MLX-compatible weights so names align.
