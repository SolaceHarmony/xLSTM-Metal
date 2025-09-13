# scripts

Utilities and production tools for running, optimizing, and debugging xLSTM on Apple Silicon (MPS).

**PYTHON NOTE (READ ME FIRST): python3 is trash - it's the MacOS python which I can't upgrade. python is the 3.12 version from conda.**

Subfolders
- build: Build helpers (e.g., extension or toolchain setup scripts).
- checks: Lightweight validators and sanity checks.
- debug: One-off probes and debug tools (MPS memory, dims, storage API).
- downloads: Model/weight download and loading helpers.
- optimizer/monitor: `optimize_mps.py`, `xltop.py`.

Legacy runners, benchmarks, and experiments have been moved under `lab/<date>-*/` to avoid confusion with the Solace packages.

Production entry (Torch + Ray)
- `xlstm_generate_pt.py` at repo root (uses Solace Torch package)
- JSON configs in `configs/` are layered automatically. Use `--print-effective-config` to inspect.

Diagrams

1) Bands × chunks scheduler (Ray default)

```
heads → bands (size = heads_per_band)
sequence → tiles (size = chunk_size)

actors A_k per band; iterate tiles sequentially per band, parallel across bands

time →
band 0:  A0(t0) → A0(t1) → A0(t2) → ...
band 1:  A1(t0) → A1(t1) → A1(t2) → ...
...
stitch outputs + states across bands per tile
```

2) Entrypoint → Backends

```
xlstm_generate_pt.py
  └─ Solace model (xLSTMTorch) with runtime_opts
      └─ compiled MPS step (metal), sequence native_sequence__metal,
         chunkwise {ray_compiled_steps|queued_compiled_steps}
```
