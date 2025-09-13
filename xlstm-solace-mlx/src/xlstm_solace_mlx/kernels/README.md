MLX Fast Kernels (mx.fast.metal_kernel)

Purpose
- Curated, production MLX/Metal kernels used by the Solace MLX path. Kernels are authored in the "body-only" style and compiled at runtime via `mx.fast.metal_kernel`.

What lives here
- gemm_kernels.py — tiled GEMM variants (AV, AT·B), device heuristics select tiles (e.g., M3 uses 8x16x32); exported builders return compiled kernels.
- qr_kernels.py — column-dot and update kernels for QR; SIMD and simple variants available.
- ivf_kernels.py — fused L2 + top‑k (single and batched) and device merges across partial lists.
- shaders.py — simple body-only examples (soft_cap, memcpy) demonstrating the contract and launch pattern.

Launch contract (summary)
- `mx.fast.metal_kernel(name, input_names, output_names, header, source, ensure_row_contiguous=True)` produces a callable that accepts `inputs=[...]`, `output_shapes=[...]`, `output_dtypes=[...]`, and launch dims `grid=(gx,gy,gz)`, `threadgroup=(tx,ty,tz)`.
- Kernels receive a `shape` buffer (uint32) to pass extents; avoid hardcoding sizes in shader code.

Runtime configuration
- See `xlstm_mlx/tools/mlx_runtime.py` and the MLX CLI.
- The CLI layers JSON: packaged `mlx_golden.json` → `configs/mlx_hardware_params.json` → optional `--profile/--config` → CLI.
- Typical tunables: QR dot mode, fast_head (tiled GEMM on final projection).

Where results are produced
- Historical MLX benchmarking scripts are under `lab/2025-09-11-benchmarks/` (migrated from scripts/benchmarks). They write CSVs/plots into `runs/benchmarks/mlx/<timestamp>/`.

