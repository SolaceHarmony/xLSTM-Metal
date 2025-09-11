Legacy Runners (migrated from scripts/runners and related entry scripts)

Scope
- Early runner scripts that target upstream namespaces or exploratory flows.

Contents
- runners/inference.py — example inference wrapper around upstream layout
- runners/run_xlstm.py — generic runner wiring
- runners/train_xlstm.py — training scaffolding (non‑production)
- runners/xlstm_quick.py — quick local runner using Ray (legacy structure)
- run_hf_xlstm_mps.py — upstream HF‑style runner
- run_hf_xlstm_metal.py — upstream HF runner (Metal experiments)
- xlstm_run.py — generic entry wrapper
- serve_xlstm.py — experimental service harness

Notes
- These are superseded by the Solace CLIs and entry points.

