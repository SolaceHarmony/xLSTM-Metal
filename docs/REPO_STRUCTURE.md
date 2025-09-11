Repository Structure (High-level)

- `mlstm_kernels/torch/`
  - `backend_module.py`: mLSTM backend wrapper and config (routes chunkwise/sequence/step).
  - `chunkwise/`: chunkwise (prefill) kernels and drivers.
    - `native/`: pure-PyTorch eager.
    - `native_compiled/`: pure-PyTorch compiled variants.
    - `queued_compiled/`: GPU-only queued compiled-step driver for MPS.
    - `metal/` and `triton_*` folders: legacy paths (Metal custom and Triton; disabled on Apple in this setup).
  - `recurrent/`: sequence/step kernels.
    - `metal/compiled.py`: compiled mLSTM step used for Apple MPS.
    - `native_sequence/`: pure-PyTorch sequence loop.
  - `parallel/`: fully-parallel (quadratic) kernels for analysis/experiments.

- `xlstm-solace-torch/src/xlstm_solace_torch/*` – Solace Torch package (model, kernels, Ray orchestration)
- `xlstm-solace-mlx/src/xlstm_solace_mlx/*` – Solace MLX package (model/components, CLI)

- `scripts/`
  - Production tools only (optimizer, monitor, downloads, checks). Legacy runners/benchmarks/experiments moved under `lab/<date>-*/`.

- `implementations/`
  - `pytorch/`, `metal/`, `mlx/`: organized legacy/reference implementations (top‑level import shims preserved).

- `research_archive/`
  - Archived experiments, notes, and demos; not wired into production runners.

Notes
- JSON runtime configs live in `configs/` and drive production settings. Torch runner layering:
  1) `configs/runtime_defaults.json`
  2) Auto-picked newest profile by backend (e.g., `*ray*.json` or `*queued*.json`), unless `--profile` is provided
  3) Optional `--config <path>`
  4) CLI flags
- See `docs/REPO_HYGIENE.md` for a production vs experiments overview.

- `tools/`
  - `test_metal_parity.py`: mLSTM parity checks.
  - `test_slstm_parity.py`: sLSTM parity checks.

Conventions
- Kernel registry keys (Ray is default chunkwise):
  - `chunkwise` backends: `"chunkwise--<variant>"`, e.g., `chunkwise--ray_compiled_steps` (default), `queued_compiled_steps` (legacy).
  - `sequence` backends: `"native_sequence__<variant>"`, e.g., `native_sequence__metal`.
  - `step` backends: `"<variant>"`, e.g., `metal`.
- Compiled backends are strict: no CPU fallback; raise on unsupported device.
