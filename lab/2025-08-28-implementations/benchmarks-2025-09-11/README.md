Benchmarks (migrated from scripts/benchmarks and related bench_*.py)

Scope
- Historical benchmarking helpers for MLX and MPS paths, including the
  “ultimate” benchmark harness and standalone MPS bench scripts.

Contents
- run_mlx_benchmarks.py — MLX model micro/throughput benchmarks
- benchmark.py — common harness used by some MLX tests
- xlstm_ultimate_benchmark.py — consolidated benchmark suite
- bench_mps.py — standalone MPS decode/prefill benchmark
- bench_mps_ray.py — MPS benchmark with Ray orchestration

Notes
- These scripts predate the Solace CLIs and package layout. Prefer the
  production CLIs and internal profilers for current measurements.

