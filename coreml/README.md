Core ML mLSTM Step (MIL)

This directory contains a hand‑assembled Core ML (MIL/MLProgram) build of a single mLSTM step cell.

Entry point
- `build_mlstm_step.py`: constructs a MIL Program for a single mLSTM step, mirroring the upstream
  equations. Inputs and outputs follow the native recurrent step:
  - Inputs: q (B,NH,DHQK), k (B,NH,DHQK), v (B,NH,DHV), i (B,NH,1), f (B,NH,1),
            c_old (B,NH,DHQK,DHV), n_old (B,NH,DHQK), m_old (B,NH,1)
  - Outputs: h (B,NH,DHV), c_new (B,NH,DHQK,DHV), n_new (B,NH,DHQK), m_new (B,NH,1)

Notes
- Built using coremltools MIL builder: matmul/add/mul/exp/log/maximum/abs/real_div/expand_dims/squeeze.
- logsigmoid(f) implemented as -log(1 + exp(-f)) to avoid custom ops.
- Shapes use a symbolic batch dimension; head dims and per‑head dims are fixed at build time.
- The resulting MIL Program can be converted to an MLProgram via coremltools conversion in a separate step.

Export
- `export_mlstm_step.py` converts the MIL Program to an MLProgram `.mlpackage` using coremltools.
  Example:
    PYTHONPATH=. python coreml/export_mlstm_step.py --dhqk 512 --dhv 512 --nh 8 --out mlstm_step.mlpackage

- `export_xlstm_decode_step.py` converts the full decode step (unrolled blocks) to a stateful MLProgram.
  Per‑block recurrent states (c,n,m) are declared as Core ML states (ct.StateType) so Core ML updates them in‑place.
  Example:
    PYTHONPATH=. python coreml/export_xlstm_decode_step.py --V 50304 --D 4096 --L 32 --NH 8 --DHQK 512 --DHV 512 --out xlstm_decode_step.mlpackage

- `export_xlstm_decode_stateful.py` builds the native MIL stateful decode (read_state/coreml_update_state)
  with a concrete block schedule (0=mLSTM, 1=sLSTM), then converts to MLProgram.
  Example:
    PYTHONPATH=. python coreml/export_xlstm_decode_stateful.py \
      --V 50304 --D 4096 --L 32 --NH 8 --DHQK 512 --DHV 512 --K 4 \
      --block-types "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" \
      --out xlstm_decode_stateful.mlpackage

Integration (Swift/Obj‑C)
- Load `mlstm_step.mlpackage` with Core ML, manage state (c,n,m) buffers between calls.
- For per-token projection or gating beyond the core step, compose small layers in Swift or build MIL for them as well.

Notes on placement
- Compute units = ALL to allow ANE/GPU/CPU. Keep ops ANE‑friendly (avoid dynamic shapes beyond batch, prefer FP16/FP32).

Compression (Core ML Tools 8)
- After export, you can apply post‑training compression (palettization/quantization/pruning) with `coremltools.optimize`.
- New per‑grouped‑channel palettization and per‑block int4 quantization can improve size/latency on ANE/GPU.

Stateful xLSTM Decode Step (MIL)
================================

Overview
--------
We construct the entire xLSTM decode step as a single MIL Program (no ONNX or TorchScript translation), unrolling L blocks and using Core ML’s native state tensors so the runtime updates state in‑place between calls. This mirrors upstream xLSTM semantics and is optimized for Apple Silicon (ANE/GPU/CPU).

Key principles
- One Program: unroll exactly L blocks in the decode step; block schedule (mLSTM vs sLSTM) is baked in at build time.
- Native States: per‑block recurrent state is declared via `StateTensorSpec`, read via `read_state`, and updated via `coreml_update_state`.
- LayerNorm everywhere: upstream uses LayerNorm (not RMSNorm). We use `mb.layer_norm` before the block and before the FFN.
- Causal conv ring buffer: decode uses a short causal conv front‑end (kernel size K) with a small per‑block ring buffer state.
- Outputs: only logits; states are kept internally and updated in place by Core ML.

Per‑block anatomy
-----------------
1) Pre‑norm: `x_norm = layer_norm(x, gamma, beta)`.

2) Causal conv (decode step):
   - Keep a state `conv_state_i` of shape `(B, K-1, D)`.
   - Form window = concat(conv_state_i, x) along time.
   - Compute conv: `y_conv = sum_t (w[t] * window[:, t, :])` with MIL ops (concat/slice/mul/add).
   - Update `conv_state_i` by shifting and appending `x` via `coreml_update_state`.
   - Branch usage:
     - mLSTM: use `y_conv` for q/k projections; v and all gates use `x_norm` (matches upstream mLSTMLayer).
     - sLSTM: use `y_conv` for i/f; z and o from `x_norm` (matches upstream sLSTMLayer).

3) Projections and gates (head‑wise):
   - mLSTM:
     - q_flat = `Wq * y_conv + bq`, k_flat = `Wk * y_conv + bk`, v_flat = `Wv * x_norm + bv`.
     - i_raw = `Wi * x_norm + bi`, f_raw = `Wf * x_norm + bf`, o_flat = `Wo * x_norm + bo`.
     - reshape to `(B, NH, DH)`.
   - sLSTM:
     - raw = Wx + Ry + b, split into i/f/z/o (implemented as head‑wise expand; we model as flat linears then reshape).

4) Recurrent step math (inline):
   - mLSTM (per upstream):
     - `f_log = -log(1+exp(-f_raw))`, `m_new = max(i_raw, m_old + f_log)`
     - `F_act = exp(f_log + m_old − m_new)`, `I_act = exp(i_raw − m_new)`
     - `q_s = q * (DHQK**−0.5)`
     - `c_new = F_act * c_old + I_act * (k @ v)` (outer product per head)
     - `n_new = F_act * n_old + I_act * k`
     - `h_num = (q_s @ c_new)`; `qn = (q_s dot n_new)`; `h_denom = max(abs(qn), exp(−m_new)) + eps`
     - `h_heads = h_num / h_denom` (per head)
   - sLSTM (per upstream vanilla kernel):
     - `f_log = -log(1+exp(-f_raw))`, `logfplusm = m_old + f_log`, `m_new = max(i_raw, logfplusm)`
     - `igate = min(exp(i_raw − m_new), 1)`, `fgate = min(exp(logfplusm − m_new), 1)`
     - `c_new = fgate * c_old + igate * tanh(z_raw)`; `n_new = fgate * n_old + igate`
     - `y_new = sigmoid(o_raw) * (c_new / n_new)`

5) Out path and residual:
   - Merge heads: reshape `h_heads` (mLSTM) or `y_new` (sLSTM) to `(B, NH*DHV)`.
   - LayerNorm over merged vector.
   - Out gate: `sigmoid(o_flat)` (mLSTM) or use `sigmoid(o_raw)` in sLSTM.
   - Project to model dim: `y = h_norm * sig(o) → W_out → + b_out`.
   - Residual: `x = x + y`.

6) FFN (gated, SiLU):
   - `x_ffn = layer_norm(x)`
   - `up_gate = Wup_gate * x_ffn + b`, `up_z = Wup * x_ffn + b`
   - `ff = Wdown * (SiLU(up_gate) * up_z) + b`
   - Residual: `x = x + ff`.

7) State updates (per block):
   - `coreml_update_state` for `c_i, n_i, m_i, conv_state_i` with the new tensors.

8) Head: `logits = x @ lm_W` (optional `soft_cap` if needed).

MIL program structure
---------------------
- Use the builder’s `@mb.program` with `input_specs=[TensorSpec(...), StateTensorSpec(...), ...]`.
- Read and write states with `mb.read_state` and `mb.coreml_update_state` exactly once per block.
- Bake LayerNorm parameters and all weights into MIL constants for a self‑contained MLProgram, or pass them as inputs and later freeze via a small script.

Export
------
- Convert to MLProgram with:
  ```python
  mlmodel = ct.convert(prog, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.ALL)
  ```
- Since states are native MIL states, you do NOT pass `ct.StateType` at conversion; Core ML recognizes and manages them in place.

Reset and multi‑function
------------------------
- Consider a second MIL function that zeroes all states (conv ring buffers and c/n/m) and expose it via the Core ML multi‑function API. This provides a clean “reset” entrypoint.

Compression
-----------
- After export, use `coremltools.optimize` to apply:
  - Per‑grouped‑channel palettization (ANE‑friendly) with configurable group size.
  - Per‑block int4 linear quantization (GPU‑friendly) with block size.
  - Sparsity + palettization/quantization stacking where applicable.

Integration (Swift/Obj‑C)
-------------------------
- Load the stateful MLProgram once; call it per token with the new token id. Core ML maintains and updates the recurrent states internally.
- For best results, keep precision FP16/FP32 and choose `.all` compute units; let the runtime schedule on ANE/GPU/CPU.
