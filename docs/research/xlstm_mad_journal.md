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

Notes
- No shims, no try/except fallbacks. Fail fast during study.
- Keep public API canonical (xLSTMLarge / xLSTMLargeConfig). Document deltas here as we iterate.
