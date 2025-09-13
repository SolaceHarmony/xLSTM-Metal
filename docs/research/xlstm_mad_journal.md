Title: xLSTM ↔ MAD Integration Journal

Context
- Repo: Apple‑optimized xLSTM (Torch MPS + MLX), JSON‑first runtime.
- Goal: Blend upstream xLSTM architecture controls with a MAD‑style design loop; keep canonical API.
- External reference cloned for study: /Volumes/emberstuff/Projects/mad-lab

2025-09-13 — Initial Diff & MAD framing
- Block split (upstream): mLSTMBlock = sequence mixer only; sLSTMBlock = sequence mixer + gated FFN. FFN frequency is controlled by a striping schedule.
- Our Torch block: always includes FFN after the sequence mixer. This increases compute/params per depth and changes residual/gradient shaping.
- Norm policy: upstream uses LayerNorm (pre‑norm per branch, post‑stack LN). Our path uses RMSNorm for branch norms and optional post‑stack RMSNorm.
- FFN details: upstream uses gated FFN (two‑linear up with activation, linear down), dropout, and specific init (small_init on in‑proj, Wang init on down‑proj with depth awareness). Our FFN uses SiLU gating, no dropout by default, and PyTorch default init.
- mLSTM inner path: upstream layer includes learnable skip on the inner branch and a short causal conv pre‑activation mixed back before z‑gating; we rely on kernel backends and do the residual at the block level.
- State contract: upstream step() returns (x, {mlstm_state, conv_state}); our forward optionally returns (x, state) and state is dict[int] → (c, n, m). Unification needed at the builder boundary.

Implications
- To match upstream semantics and enable MAD striping, the builder should instantiate mLSTM‑only and sequence+FFN blocks by schedule, not hard‑wire FFN in every block.
- Keep norm/init policy per profile consistent (canonical LN + upstream FFN init vs RMSNorm path) so comparisons reflect topology, not drift from numerics.
- The “total state dimension” (heads × head_state_dim) is the primary capacity knob; MAD profiles can express iso‑state designs.

Next Steps (proposed)
- Define a minimal, explicit block schedule in config (e.g., pattern or ratio) and a norm policy toggle (LN vs RMSNorm).
- Add a small MAD harness (in‑context recall, selective copy) for quick capability fingerprints of schedules.
- Unify top‑level forward/step state schema at the builder seam; adapt per backend internally.

2025-09-13 — Config hook for FFN frequency (Torch path)
- Added optional `ffn_blocks` to `xLSTMTorchConfig` and threaded block index into `mLSTMBlock`.
- If `ffn_blocks` is None: behavior unchanged (FFN in every block).
- If `ffn_blocks` is a set of indices: only those blocks include FFN; others skip the FFN branch entirely.
- This approximates upstream’s ability to control channel‑mixer frequency via striping without introducing new block types yet (sLSTM vs mLSTM).
- No shims or try/except; failures will surface if misused.

2025-09-13 — Upstream-first builder surface
- Surveyed upstream `xlstm/` tree: two parallel implementations exist:
  - `xlstm/blocks/*` + `xlstm_block_stack.py` (hybrid mLSTM/sLSTM with striping and LN).
  - `xlstm/xlstm_large/*` (mLSTM-only blocks with FFN per block and RMSNorm).
- Added `xlstm/profile_loader.py` to construct `xLSTMBlockStackConfig` and `xLSTMLMModelConfig` from plain dict/JSON.
  - Supports `slstm_at` mapping (list or "all"), optional feedforward spec, and basic mLSTM/slSTM config fields.
  - No fallbacks; missing required fields raise immediately.
- Intent: drive MAD-style profiles directly against upstream block stack without introducing aliases; keep API canonical.

2025-09-13 — Minimal MAD harness (seed)
- Added scripts to exercise a simple MAD-style task without external deps:
  - `scripts/mad/tasks.py`: in-context recall generator + accuracy with ignore index.
  - `scripts/mad/profile_run.py`: loads a JSON profile (via `xlstm/profile_loader.py`), builds `xLSTMLMModel`, runs a one-shot evaluation on the task, prints accuracy.
- Policy adherence: no try/except, fail on invalid profiles; no shims.
- Next: expand tasks (selective copying), add a short training loop for capability fingerprints (minutes, not hours), and plug in schedule experiments.

2025-09-13 — Training harness (minutes-scale)
- Added `scripts/mad/train_profile.py`: simple AdamW trainer over synthetic batches.
  - Supports tasks: in-context recall (train/eval variants) and selective copying (ignore-index targets).
  - Uses CrossEntropyLoss with ignore_index, aligns with MAD’s loss semantics.
  - Prints loss/accuracy snapshots; no silent fallbacks.
- Extended `scripts/mad/tasks.py` with `generate_selective_copying` mirroring MAD’s generator.
- Note: upstream sLSTM backend defaults to CUDA; profiles using sLSTM should be avoided on CPU until we decide on a CPU path or enforce mLSTM-only for capability probes.

2025-09-13 — TorchScript path (Torch side)
- Added TorchScript-friendly typed state and entrypoints to Torch model:
  - `forward_with_state(x, state: List[Optional[(c,n,m)]]) -> (logits, state)`
  - `generate_greedy(prefill, max_len)`
- Implemented `xLSTMBlockStack.forward_with_state` with fixed-length list state.
- TS compile/test harness: `scripts/ts/compile_and_test.py` scripts a small config and runs a forward, with explicit `set_num_threads`/`set_num_interop_threads` knobs.
- TS profile guidance: use native ATen backends (`chunkwise--native_compiled_autograd`, `native_sequence__native`, `step=native`) on MPS/CPU for scripting.
- Next: optional `_fork/_wait` in projection/gate stanzas for inter-op parallelism, then microbench thread sweeps recorded here.

Notes
- No shims, no try/except fallbacks. Fail fast during study.
- Keep public API canonical (xLSTMLarge / xLSTMLargeConfig). Document deltas here as we iterate.
