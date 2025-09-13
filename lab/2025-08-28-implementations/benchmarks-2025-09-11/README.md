Benchmarks (curated, migrated from scripts/benchmarks and related bench_*.py)

Scope
- Historical benchmarking helpers for MLX and MPS paths, including the
  “ultimate” benchmark harness and standalone MPS bench scripts.

Contents (how to interpret)
- run_mlx_benchmarks.py — MLX model micro/throughput benchmarks. Writes `runs/benchmarks/mlx/<ts>/mlx_benchmarks.csv` and charts. Exercises kernels in `xlstm_mlx/kernels/*` (mx.fast).
- benchmark.py — common harness for MLX/PyTorch comparisons; tabulates throughput as "Throughput (tok/s)" and saves CSV.
- xlstm_ultimate_benchmark.py — consolidated suite across sizes/devices; contains tuned block variants for fair comparisons.
- bench_mps.py — standalone MPS prefill/decode benchmark for Torch.
- bench_mps_ray.py — MPS benchmark with Ray orchestration; prints prefill/decode tok/s per trial.

Notes
- These are the canonical record of perf evidence. They were used to produce the `runs/benchmarks/*` artifacts with decode/prefill tok/s.
- For fresh runs today, prefer Solace CLIs and refer to `xlstm_mlx/kernels/README.md` for the current fast path; keep output CSVs/plots under `runs/benchmarks/<...>` for continuity.
- HeadLinear vs BlockLinear (MLX)
  - See `mlx_headlinear_vs_blocklinear.py` for parity and timing between:
    - Big GEMM (D→V) vs per‑head GEMMs vs block‑diagonal build + GEMM
  - Observed (example runs):
    - max|Δ| ≈ 0.0 across methods (numerical parity)
    - Big GEMM was consistently a touch faster than per‑head GEMMs; block‑diag build cost was small but non‑zero.
  - Decision: production MLX head uses a single big GEMM (our tiled `gemm_av`) with weight (V,D) — see `xLSTMMLX.FastHead`.
