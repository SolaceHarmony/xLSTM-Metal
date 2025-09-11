# Agents Guide: Operating the Solace xLSTM Fork (Torch + MLX)

This guide is for humans and automation agents working in this repo. It explains what the project is, how to run it safely on Apple Silicon, where the production entrypoints live, and how runtime configuration works now (JSON‑first, no envs in production).

Assumptions: macOS on Apple Silicon, conda env `base` (Python 3.12). CUDA is not a supported target for the Torch path; MLX may support CUDA on some platforms but is not a primary focus here.

## What This Project Is
- Goal: High‑throughput xLSTM (mLSTM + sLSTM) inference on Apple Silicon via compiled PyTorch MPS backends (Torch) and a pure MLX path (MLX).
- Core: Small compiled kernels scheduled across head‑bands and sequence tiles; all math on GPU.
- Prefill backends (Torch):
  - `chunkwise--ray_compiled_steps` — Ray actors coordinate compiled step kernels (default).
  - `chunkwise--queued_compiled_steps` — CPU thread pool schedules compiled step kernels.
  - `chunkwise--native_compiled_autograd` — pure compiled native comparator.
- Decode (Torch): compiled step kernel in a loop (`step_kernel="metal"`; `sequence_kernel="native_sequence__metal"`).

## Production Entrypoints
- Torch (MPS + Ray):
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=.:xlstm-solace-torch/src \
  conda run -n base python xlstm_generate_pt.py \
    --model_path ./xlstm_7b_model \
    --prompt "The capital of France is" \
    --max_new_tokens 32
  ```
  - Inspect merged runtime: add `--print-effective-config`.

- MLX (no Ray):
  ```bash
  PYTHONPATH=.:xlstm-solace-mlx/src conda run -n base python -m xlstm_solace_mlx.cli \
    --prompt "Hello" --max_new_tokens 16 --profile mlx_golden --print-config
  ```

## JSON Runtime Profiles (No Envs in Production)
- Config files live under `./configs/` and inside the packages. Layering order (lowest → highest):
  1) `configs/runtime_defaults.json` (Torch) or `configs/mlx_hardware_params.json` (MLX)
  2) Packaged “golden” profile: `xlstm_solace_torch/configs/golden_{ray|queued}.json` (Torch) or `xlstm_solace_mlx/configs/mlx_golden.json` (MLX)
  3) Auto-picked newest matching profile in `./configs` (Torch only; if no packaged golden or when `--profile` not given)
  4) Optional `--profile <name>` (looks in `./configs`)
  5) Optional `--config <path>`
  6) CLI flags
- The merged settings are passed as `runtime_opts` to the model and kernels. Avoid envs.
- See `configs/README.md` for details and optimizer integration.

## Tools At A Glance
- `scripts/optimize_mps.py`: random/GA sweeps (writes `runs/mps_opt/<run>/` + `best.json`). The Torch entry auto‑applies best.json chunk_size/heads_per_band unless overridden.
- `scripts/xltop.py`: terminal monitor with TUI and machine‑friendly modes
  - Curses UI: `conda run -n base python scripts/xltop.py`
  - Polling: `--no-curses --poll 1.0`; JSON: `--json`; NDJSON stream: `--json-stream --poll 0.7`
  - Stdin control (for agents): `--stdin-commands` then send lines: `kill <pid>`, `ray stop`, `empty_cache`, `interval <sec>`, `quit`
- Ray CLI: `ray status`, `ray list actors`, `ray memory`, `ray stop --force`

## Memory Watchdog & Telemetry (Torch)
- Module: `xlstm_solace_torch.kernels.torch.monitoring.memory`
  - Samples process RSS, system available/total, and `torch.mps` allocated/reserved.
  - Optional CSV logging via the runner flag `--mem-log`.
  - Thresholds: percentage of total or absolute MB; soft actions and hard abort.
- Defaults tuned for safety (UMA): soft 85%, hard 92%, poll 200 ms.
- Configure via JSON (`runtime_opts.mem_watchdog`) or CLI flags; envs are avoided in production.

## Ray Lifecycle & Dashboard (Torch)
- Local mode by default; dashboard optional.
  - `--ray-local-mode {0|1}`; `--ray-dashboard [--ray-dashboard-port 8265]`; `--ray-keep-alive`.
  - Actors are terminated on exit; auto‑shutdown if runner started Ray (unless `--ray-keep-alive`).
  - Cleanup if needed: `ray stop --force`.

## Safety & Cleanup
- Enforce GPU‑only: `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- If a run goes rogue: `kill -TERM <pid>` (then `-KILL`). xltop TUI has a `K` hotkey.
- Debug memory: `scripts/xltop.py --json` (rss_mb, mps_alloc_mb); `/usr/bin/vmmap -summary <pid>` as fallback.

## Repo Layout (Solace Fork)
- Solace Torch package: `xlstm-solace-torch/src/xlstm_solace_torch/*` (model, kernels, Ray orchestration, packaged configs)
- Solace MLX package:  `xlstm-solace-mlx/src/xlstm_solace_mlx/*` (model/components, CLI, packaged configs)
- Production tools:    `scripts/` (optimizer, monitor, downloads, checks)
- Legacy/experiments:  `lab/<date>-*/` (benchmarks, legacy runners, experiments)

## Notes for Agents (Automation)
- Use conda base: `conda run -n base python …`.
- For Torch runs, set `PYTORCH_ENABLE_MPS_FALLBACK=0` and add `PYTHONPATH=.:xlstm-solace-torch/src` (or install the package and use console scripts).
- Prefer JSON‑first workflow; avoid relying on envs except for isolated debugging.

Further reading:
- `configs/README.md` — how runtime presets layer and how to freeze a golden profile
- `docs/TUNING_GUIDE.md` — background on chunk_size, heads_per_band, workers, streams
- `docs/APPLE_MPS_GUIDE.md` — platform specifics and best practices
