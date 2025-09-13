Runtime Profiles and Presets (Production)

Purpose
- This folder contains JSON presets that drive the production runtime for the Solace Torch (Ray) and Solace MLX paths.
- The Torch entry auto-selects a preset if none is specified; MLX can also layer a profile.

Files
- runtime_defaults.json — base defaults applied first (Torch).
- max_mps.json — hardware-centric limits for MPS (reference).
- mlx_hardware_params.json — hardware-centric choices for MLX (GEMM tiling, QR mode, etc.).
- varquant_default.json — presets for variable-quant experiments (non-production).
Note: Optimizer configs now live under `scripts/optimizer/configs/`.

How profiles are applied (Torch)
- Layering order (lowest → highest):
  1) runtime_defaults.json
  2) Auto-picked profile by backend: newest JSON whose name includes "ray" (for Ray) or "queued" (for queued)
  3) --config <path> (optional)
  4) CLI flags
- The merged settings are passed to the model/kernels via `runtime_opts`; production avoids environment variables.
- Inspect the final, merged view with:
  PYTHONPATH=.:xlstm-solace-torch/src python xlstm_generate_pt.py --print-effective-config

How profiles are applied (MLX)
- Base defaults from `mlx_hardware_params.json`, then an optional `--profile <name>` and `--config <path>`, then CLI.
- Inspect the merged view with:
  PYTHONPATH=.:xlstm-solace-mlx/src python -m xlstm_mlx.cli --profile mlx_hardware_params --print-config

Optimizer integration (Torch)
- The optimizer writes sweeps under `runs/mps_opt/<timestamp>/` and a `best.json` summarizing the best settings found (e.g., chunk_size, heads_per_band).
- The Torch entry auto-applies the latest `runs/mps_opt/*/best.json` values for chunk_size/heads_per_band unless explicitly overridden.
- Packaged goldens:
  - `xlstm_torch/configs/golden_ray.json` derives from `scripts/optimizer/configs/experiment_ray16k.json` (workers=4, heads_per_band=4, chunk_size=32) with safe memory defaults.
  - `xlstm_torch/configs/golden_queued.json` mirrors stable queued settings (workers=6, heads_per_band=4, chunk_size=32) with safe memory defaults.
- To force a specific local preset instead: pass `--profile <name>` (without `.json`).

Tips
- Keep presets small and focused: scheduling knobs (e.g., chunk_size, heads_per_band, workers, streams), Ray lifecycle flags, memory watchdog knobs.
- Avoid env vars in presets; production runners will pass options via model `runtime_opts` to the kernels and drivers.
