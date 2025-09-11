# MLX Runtime Configuration (No Envs Required)

Use `tools/mlx_runtime.py` to configure MLX kernel behavior programmatically. This avoids environment variables and keeps settings local to the code invoking kernels.

## APIs

```
from tools.mlx_runtime import (
  configure_gemm, configure_qr, configure_ivf, configure_model,
  get_runtime_config, reset_runtime_config,
)
```

- `configure_gemm(pad: bool|None, align_execw: bool|None, double_buffer: bool|None)`
  - `pad`: add `+1` padding on the second tile dimension to mitigate shared‑memory conflict patterns.
  - `align_execw`: align square tile size `T` to the device `threadExecutionWidth` when `T*T ≤ 1024`.
  - `double_buffer`: enable double‑buffered tiles (ping‑pong) to prefetch tile `t+1` while computing tile `t`.
- `configure_qr(dot_mode: "auto|simd|simple"|None)`
  - Select the projection kernel (SIMD warp‑reduction vs simple per‑column loop).
- `configure_ivf(tpb: int|None)`
  - Threads per threadgroup for IVF top‑k kernels when scanning lists.
- `configure_model(fast_head: bool|None)`
  - Toggle the xLSTM fast projection head (tiled GEMM) at runtime.
- `get_runtime_config()` returns the current settings; `reset_runtime_config()` clears all overrides.

## Example

```
from tools.mlx_runtime import configure_gemm, configure_qr, configure_model
from src.mlx_impl.xlstm_mlx import create_xlstm_model

# Prefer runtime config to envs
configure_gemm(pad=True, align_execw=True, double_buffer=True)
configure_qr(dot_mode="simd")
configure_model(fast_head=True)

model = create_xlstm_model(
    vocab_size=32000, num_layers=16, signature=(1,1),
    inp_dim=1536, head_dim=128, head_num=12, dropout=0.0,
)
```

## CLI Integration (Solace MLX)

Use the Solace MLX CLI which layers JSON profiles from `configs/`:

```bash
PYTHONPATH=.:xlstm-solace-mlx/src python -m xlstm_solace_mlx.cli \
  --prompt "Hello" --max_new_tokens 16 \
  --profile mlx_hardware_params --print-config
```

Profile layering: `configs/mlx_hardware_params.json` → optional `--profile` JSON → optional `--config` JSON → CLI flags.

## Precedence

1. Runtime config (this module)
2. Environment variables (fallback)
3. Device‑aware tuning JSON (defaults; `tools/mlx_tuning.py`)

Runtime config is the recommended way to manage kernel behavior in this repo.
