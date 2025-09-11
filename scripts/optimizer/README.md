Optimizer Pipeline (Solace Torch)

Scope
- Automated search and evaluation of runtime parameters (chunk_size, heads_per_band, workers) for the Torch MPS path.
- Uses local HF-style checkpoints and runs on Apple Silicon.

Scripts
- optimize_mps.py — random/GA sweeps; writes `runs/mps_opt/<run>/` with `summary.csv`, `trials.jsonl`, `best.json`.
- save_outputs_for_trials.py — regenerates greedy continuations per CSV row into `<run>/outputs/` with filenames encoding params.
- judge_outputs.py — scores outputs (avg_logprob, perplexity, distinct-2/3) and writes `ratings.jsonl`/`ratings.csv`.
- plot_opt_results.py — plots speed vs settings and summaries from `summary.csv` and ratings.

Usage (Ray backend, GA)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=.:xlstm-solace-torch/src \
  conda run -n base python scripts/optimizer/optimize_mps.py \
    --backend ray --config scripts/optimizer/configs/experiment_ray16k.json \
    --model_path ./xlstm_7b_model --new 64 --mode ga --generations 5 --population 10 --repeats 1

RUN=scripts/optimizer/runs/ray_YYYYMMDD_HHMMSS_tag
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=.:xlstm-solace-torch/src \
  conda run -n base python scripts/optimizer/save_outputs_for_trials.py \
    --run $RUN --model_path ./xlstm_7b_model \
    --prompt-file ./prompts/long_form.txt --new 32 \
    --outputs $RUN/outputs

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=.:xlstm-solace-torch/src \
  conda run -n base python scripts/optimizer/judge_outputs.py \
    --model_path ./xlstm_7b_model \
    --prompt-file ./prompts/long_form.txt \
    --outputs $RUN/outputs

conda run -n base python scripts/optimizer/plot_opt_results.py --run $RUN

# Curated best snapshots:
# scripts/optimizer/best/ray_latest.json and timestamped copies are updated on improvements.
```

Notes
- The optimizer scripts import model helpers from `xlstm_generate_pt.py` and the Solace Torch package; they no longer rely on upstream namespaces.
- For production runs, prefer the packaged golden profiles under `xlstm_solace_torch/configs/` and the JSON-first entry (`xlstm_generate_pt.py`).
