Metal Implementations (high‑performance lineage)

This directory contains the MPS/Metal implementations that led to the current Solace Torch and Solace MLX production paths. Files include TorchScript‑on‑MPS models, eager/compiled MPS variants, and Metal kernel experiments. These were benchmarked and produced perf artifacts in tandem with the MLX fast kernels.

Files and purpose
- `xlstm_jit_metal.py` — TorchScript on MPS (jit.trace, optimize_for_inference); TorchScript‑friendly layers (soft cap, RMSNorm). Used for early throughput baselines and parity.
- `xlstm_jit_simple.py` — Simplified TorchScript path used for smoke/perf validation.
- `xlstm_metal_complete.py` — Integrated Metal variant prior to Solace refactor; unified model with MPS‑tuned ops.
- `xlstm_metal_optimized.py` — Eager/compile MPS optimizations (gating, projection layout) that informed the Solace compiled step.
- `xlstm_unified_metal.py` — Unified model with a structured config and backend selection, used in targeted MPS experiments.
- `xlstm_metal_kernels.py`, `xlstm_metal_hpc_limb*.py` — Metal kernel experiments and high‑precision limb variants that inspired the MLX fast metal kernels in the Solace MLX package.

Relation to production
- Torch (MPS): these led to the compiled step in `xlstm_torch.kernels.torch.recurrent.metal.compiled` + `native_sequence__metal` loop + Ray/queued schedulers.
- MLX: these informed the current MLX fast path in `xlstm_mlx/kernels/*` where kernels are authored via `mx.fast.metal_kernel`.

Benchmarks and outputs
- Related benchmark harnesses are co-located here under `../benchmarks-2025-09-11/` (migrated). They produced the decode/prefill tok/s CSVs/plots under `runs/benchmarks/*`.
- Use those scripts to reproduce performance, and compare to today’s Solace CLIs.
