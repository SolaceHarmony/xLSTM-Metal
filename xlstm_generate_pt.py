
"""
Run a local xLSTM HF checkpoint on Apple Silicon (MPS) using the compiled backends.

- Loads HF-style sharded safetensors from a local directory (config.json + model-*.safe        def _load_packaged_profile(backend):
            import importlib.resources as ir
            pkg = 'xlstm_torch.configs'
            fname = 'golden_ray.json' if 'ray' in backend else 'golden_queued.json'
            with ir.files(pkg).joinpath(fname).open('r') as f:
                import json as _json
                return _json.load(f)- Instantiates Solace xLSTMTorch with compiled kernels on MPS.
- Maps known key differences (backbone.embeddings.weight -> embedding.weight).
"""
import argparse
import os
import json
from pathlib import Path
import signal

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
import torch.nn.functional as F

from xlstm_torch.models.model import xLSTMTorch, xLSTMTorchConfig

# Global abort flag to allow SIGTERM-triggered graceful stop during decode
_ABORT_FLAG = {"stop": False}
from xlstm_torch.kernels.torch.monitoring.memory import MemoryMonitor
from xlstm_torch.kernels.torch.monitoring.ray_metrics import make_gauges


def load_local_config(config_path: Path) -> xLSTMTorchConfig:
    """Load HF config.json and build xLSTMTorchConfig.

    - Copies all relevant fields from JSON.
    - On MPS, forces compiled MPS backends (Ray, metal, native_sequence__metal).
      Other fields (e.g., chunk_size) honor JSON unless CLI overrides are given.
    """
    cfg = json.loads(config_path.read_text())

    # Base construction from JSON
    mcfg = xLSTMTorchConfig(
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        vocab_size=cfg["vocab_size"],
        use_bias=cfg.get("use_bias", False),
        norm_eps=cfg.get("norm_eps", 1e-6),
        norm_reduction_force_float32=cfg.get("norm_reduction_force_float32", True),
        add_out_norm=cfg.get("add_out_norm", True),
        qk_dim_factor=cfg.get("qk_dim_factor", 0.5),
        v_dim_factor=cfg.get("v_dim_factor", 1.0),
        gate_soft_cap=cfg.get("gate_soft_cap", 15.0),
        output_logit_soft_cap=cfg.get("output_logit_soft_cap", 30.0),
        weight_mode=cfg.get("weight_mode", "single"),
    )

    # Optional tunables that should mirror JSON unless overridden later
    if "chunk_size" in cfg:
        mcfg.chunk_size = int(cfg["chunk_size"])  # JSON default often 64
    if "mode" in cfg:
        mcfg.mode = cfg["mode"]
    if "return_last_states" in cfg:
        mcfg.return_last_states = bool(cfg["return_last_states"])
    if "autocast_kernel_dtype" in cfg:
        mcfg.autocast_kernel_dtype = cfg["autocast_kernel_dtype"]
    if "inference_state_dtype" in cfg:
        mcfg.inference_state_dtype = cfg["inference_state_dtype"]
    if "ffn_proj_factor" in cfg:
        mcfg.ffn_proj_factor = float(cfg["ffn_proj_factor"])
    if "ffn_round_up_to_multiple_of" in cfg:
        mcfg.ffn_round_up_to_multiple_of = int(cfg["ffn_round_up_to_multiple_of"])

    # Backends: on MPS, force compiled defaults (ignore Triton in JSON)
    if torch.backends.mps.is_available():
        # Ray by default unless explicitly overridden via env
        default_chunkwise = os.environ.get("XLSTM_CHUNKWISE_BACKEND", "ray_compiled_steps")
        mcfg.chunkwise_kernel = f"chunkwise--{default_chunkwise}"
        mcfg.sequence_kernel = "native_sequence__metal"
        mcfg.step_kernel = "metal"
        # Sensible inference defaults
        mcfg.mode = "inference"
        mcfg.return_last_states = True
        if "autocast_kernel_dtype" not in cfg:
            mcfg.autocast_kernel_dtype = "bfloat16"
        if "inference_state_dtype" not in cfg:
            mcfg.inference_state_dtype = "float32"
        # Do not set envs; runtime opts will carry defaults.
    else:
        # Non-MPS: respect JSON kernels if present
        if "chunkwise_kernel" in cfg:
            mcfg.chunkwise_kernel = cfg["chunkwise_kernel"]
        if "sequence_kernel" in cfg:
            mcfg.sequence_kernel = cfg["sequence_kernel"]
        if "step_kernel" in cfg:
            mcfg.step_kernel = cfg["step_kernel"]

    return mcfg


def load_local_weights(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    idx = json.loads(index_path.read_text())
    files = sorted({f for f in idx["weight_map"].values()})
    state = {}
    for f in files:
        fp = model_dir / f
        with safe_open(str(fp), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                state[k] = sf.get_tensor(k)
    # Key mapping
    if "backbone.embeddings.weight" in state:
        state["embedding.weight"] = state.pop("backbone.embeddings.weight")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./xlstm_7b_model")
    ap.add_argument("--config", type=str, default=None, help="Optional JSON file of runtime overrides (path)")
    ap.add_argument("--profile", type=str, default=None, help="Profile name under ./configs (e.g., experiment_ray16k, tunables_ray16k)")
    ap.add_argument("--prompt", type=str, default=None, help="Inline prompt text (use --prompt-file for long context)")
    ap.add_argument("--prompt-file", type=str, default=None, help="Path to a prompt text file")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    # Optional overrides; if omitted, JSON+MPS defaults are used
    ap.add_argument("--workers", type=int, default=None, help="CPU coordinator threads (queued backend)")
    ap.add_argument("--heads-per-band", type=int, default=None, help="Heads per task (MPS queued/Ray scheduler)")
    ap.add_argument("--streams", type=int, default=None, help="MPS streams (queued backend; defaults to workers)")
    ap.add_argument("--chunk-size", type=int, default=None, help="Chunk size for compiled backends (defaults from JSON)")
    ap.add_argument("--chunkwise-backend", type=str, default=None,
                    help="Override chunkwise backend key (ray_compiled_steps, queued_compiled_steps, native_compiled_autograd)")
    ap.add_argument("--stats-log", type=str, default=None, help="Optional CSV file to log rolling decode stats")
    ap.add_argument("--stats-every", type=int, default=1, help="Log every N decode steps to stats log")
    # Memory monitoring / watchdog (global logging; drivers also have watchdogs)
    ap.add_argument("--mem-log", type=str, default=None, help="Write process/MPS memory CSV (ts,rss,avail,total,mps_alloc,mps_reserved)")
    ap.add_argument("--mem-every", type=int, default=None, help="Memory sampling period in ms (default 200)")
    ap.add_argument("--mem-soft-pct", type=float, default=None, help="Soft memory threshold as fraction of total (default 0.85)")
    ap.add_argument("--mem-hard-pct", type=float, default=None, help="Hard memory threshold as fraction of total (default 0.92)")
    ap.add_argument("--mem-soft-mb", type=float, default=None, help="Soft memory threshold in MB (overrides pct)")
    ap.add_argument("--mem-hard-mb", type=float, default=None, help="Hard memory threshold in MB (overrides pct)")
    ap.add_argument("--mem-action", type=str, default=None, help="Comma actions on soft limit: warn,empty_cache (default warn,empty_cache)")
    # CfC logit calibrator (experimental; default off)
    ap.add_argument("--cfc-calibrate", type=str, default="off", choices=["off","sigmoid","lecun_tanh"], help="Apply CfC-based per-step logit calibration")
    ap.add_argument("--cfc-hidden", type=int, default=32, help="CfC calibrator hidden size")
    ap.add_argument("--cfc-backbone", type=int, default=64, help="CfC calibrator backbone units")
    ap.add_argument("--cfc-mode", type=str, default="default", choices=["default","no_gate","pure"], help="CfC core mode")
    ap.add_argument("--cfc-topk", type=int, default=0, help="If >0, apply sparse bias over top-K tokens from CfC hidden")
    # Ray / Dashboard controls
    ap.add_argument("--ray-dashboard", action="store_true", help="Start a local Ray head with dashboard enabled (http://127.0.0.1:8265)")
    ap.add_argument("--ray-dashboard-port", type=int, default=8265, help="Ray dashboard port (default 8265)")
    ap.add_argument("--ray-local-mode", type=int, default=None, choices=[0,1], help="Set Ray local_mode explicitly (1=in-process, 0=multiprocess)")
    ap.add_argument("--ray-keep-alive", action="store_true", help="Keep Ray head running after run (dashboard stays up)")
    # Stopping criteria
    ap.add_argument("--stop-string", action="append", default=None, help="Stop when any of these strings appears in the generated text (repeat flag to add multiple)")
    ap.add_argument("--no-eos-stop", action="store_true", help="Do not stop on tokenizer.eos_token_id")
    # Dry-run / visibility
    ap.add_argument("--print-effective-config", action="store_true", help="Print the effective runtime + backend config and exit")
    args = ap.parse_args()

    # JSON configuration loader and overlay
    def _load_json(fp):
        import json as _json
        from pathlib import Path as _P
        p = _P(fp)
        if not p.exists():
            return {}
        return _json.loads(p.read_text())

    def _apply_runtime_overrides(args_obj):
        """Apply JSON runtime defaults and profiles into args/env/mcfg.
        Precedence (lowest→highest): runtime_defaults.json → profile → --config → CLI
        """
        # Base defaults
        base = _load_json("configs/runtime_defaults.json")
        # Named profile under package or ./configs, or auto-pick latest matching backend if not provided
        prof = {}
        from pathlib import Path as _P
        cfg_dir = _P("configs")
        # Helper: load packaged golden profile
        def _load_packaged_profile(backend: str) -> dict:
            import importlib.resources as ir
            pkg = 'xlstm_torch.configs'
            fname = 'golden_ray.json' if 'ray' in backend else 'golden_queued.json'
            with ir.files(pkg).joinpath(fname).open('r') as f:
                import json as _json
                return _json.load(f)
        if args_obj.profile:
            cand = cfg_dir / f"{args_obj.profile}.json"
            prof = _load_json(str(cand)) or _load_packaged_profile(args_obj.profile)
        else:
            backend = (args_obj.chunkwise_backend or base.get("chunkwise_backend") or "ray_compiled_steps")
            # Prefer packaged golden, then newest matching in ./configs
            prof = _load_packaged_profile(backend)
            if not prof:
                patt = "*ray*.json" if "ray" in str(backend) else "*queued*.json"
                candidates = sorted(cfg_dir.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    prof = _load_json(str(candidates[0]))
        # Explicit config path
        cfg_file = _load_json(args_obj.config) if args_obj.config else {}

        # Merge dictionaries
        merged = {}
        for d in (base, prof, cfg_file):
            merged.update({k: v for k, v in d.items() if v is not None})

        # Apply into args when CLI not explicitly set
        def maybe_set(attr, key):
            if getattr(args_obj, attr, None) in (None, False):
                if key in merged:
                    setattr(args_obj, attr, merged[key])
        for a, k in [
            ("chunk_size", "chunk_size"),
            ("heads_per_band", "heads_per_band"),
            ("workers", "workers"),
            ("streams", "streams"),
            ("chunkwise_backend", "chunkwise_backend"),
            ("ray_local_mode", "ray_local_mode"),
        ]:
            maybe_set(a, k)
        if not getattr(args_obj, "ray_dashboard", False) and merged.get("ray_dashboard"):
            args_obj.ray_dashboard = bool(merged.get("ray_dashboard"))

        # Environment overlays (not used for production; kept for visibility only)
        env_map = {
            "mem_poll_ms": "XLSTM_MEM_POLL_MS",
            "mem_soft_pct": "XLSTM_MEM_SOFT_PCT",
            "mem_hard_pct": "XLSTM_MEM_HARD_PCT",
            "mem_soft_mb": "XLSTM_MEM_SOFT_MB",
            "mem_hard_mb": "XLSTM_MEM_HARD_MB",
            "mem_action": "XLSTM_MEM_ACTION",
        }
        for k, env in env_map.items():
            if k in merged and os.environ.get(env) is None:
                os.environ[env] = str(merged[k])

    # Apply runtime JSON overlays (defaults → profile → --config)
    _apply_runtime_overrides(args)

    model_dir = Path(args.model_path)
    # Load config and model (allow dry-run without model dir)
    if model_dir.is_dir():
        base_cfg_path = model_dir / "config.json"
    else:
        base_cfg_path = None
    if base_cfg_path and base_cfg_path.exists():
        mcfg = load_local_config(base_cfg_path)
    else:
        # minimal synthetic config for dry-run printing
        mcfg = xLSTMTorchConfig(
            embedding_dim=2048,
            num_heads=32,
            num_blocks=24,
            vocab_size=50257,
        )
    # Early print-and-exit for visibility: apply merged args to reflect effective config
    if args.print_effective_config:
        # Apply CLI/JSON overrides into the config view (non-destructive for run)
        if args.chunkwise_backend:
            mcfg.chunkwise_kernel = f"chunkwise--{args.chunkwise_backend}"
        if args.chunk_size is not None:
            mcfg.chunk_size = int(args.chunk_size)
        # Build effective view
        eff = {
            "chunkwise_kernel": getattr(mcfg, "chunkwise_kernel", None),
            "sequence_kernel": getattr(mcfg, "sequence_kernel", None),
            "step_kernel": getattr(mcfg, "step_kernel", None),
            "chunk_size": getattr(mcfg, "chunk_size", None),
            "mode": getattr(mcfg, "mode", None),
            "return_last_states": getattr(mcfg, "return_last_states", None),
            "autocast_kernel_dtype": getattr(mcfg, "autocast_kernel_dtype", None),
            "inference_state_dtype": getattr(mcfg, "inference_state_dtype", None),
            "env": {
                k: os.environ.get(k) for k in [
                    "XLSTM_CHUNKWISE_BACKEND",
                    "XLSTM_MPS_HEADS_PER_BAND",
                    "XLSTM_MPS_STREAMS",
                    "XLSTM_MPS_WORKERS",
                    "XLSTM_RAY_LOCAL_MODE",
                    "XLSTM_RAY_DASHBOARD",
                    "XLSTM_RAY_DASHBOARD_PORT",
                    "XLSTM_MEM_WATCHDOG",
                    "XLSTM_MEM_POLL_MS",
                    "XLSTM_MEM_SOFT_PCT",
                    "XLSTM_MEM_HARD_PCT",
                    "XLSTM_MEM_SOFT_MB",
                    "XLSTM_MEM_HARD_MB",
                    "XLSTM_MEM_ACTION",
                ]
            }
        }
        print(json.dumps(eff, indent=2))
        return

    assert model_dir.is_dir(), f"Not a directory: {model_dir}"
    assert torch.backends.mps.is_available(), "MPS not available; requires Apple Silicon."
    # Auto-apply last optimizer best.json unless disabled
    if os.environ.get("XLSTM_USE_BEST", "1") == "1":
        from pathlib import Path as _P
        import json as _json
        runs = sorted((_P("runs/mps_opt").glob("*/best.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            best = _json.loads(runs[0].read_text())
            hpb = best.get("heads_per_band") or best.get("hpb")
            csz = best.get("chunk_size") or best.get("ck")
            if hpb is not None and os.environ.get("XLSTM_MPS_HEADS_PER_BAND") is None and args.heads_per_band is None:
                os.environ["XLSTM_MPS_HEADS_PER_BAND"] = str(int(hpb))
            if csz is not None and args.chunk_size is None:
                mcfg.chunk_size = int(csz)
    # Apply optional CLI overrides (None means: keep JSON/defaults)
    if args.chunk_size is not None:
        mcfg.chunk_size = args.chunk_size
    if args.chunkwise_backend:
        os.environ["XLSTM_CHUNKWISE_BACKEND"] = args.chunkwise_backend
        # Update the config view as well
        mcfg.chunkwise_kernel = f"chunkwise--{args.chunkwise_backend}"
    model = xLSTMTorch(mcfg)
    # Load weights
    print("Loading weights ...")
    sd = load_local_weights(model_dir)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (showing up to 10):")
        for k in missing[:10]:
            print("  ", k)
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} (showing up to 10):")
        for k in unexpected[:10]:
            print("  ", k)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # Quiet parallelism warnings without envs
    from tokenizers import utils as _tok_utils  # type: ignore
    _tok_utils.enable_parallelism(False)  # type: ignore

    # Device setup
    device = torch.device("mps")
    model = model.to(device).eval()

    # No CPU fallback: code assumes MPS; no runtime toggles.

    # Prepare input
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    else:
        assert args.prompt is not None, "Provide --prompt or --prompt-file"
        prompt_text = args.prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=inputs.dtype)
        inputs = torch.cat([bos, inputs], dim=1)

    # Optional tuning knobs for queued/Ray schedulers (no envs; pass via runtime_opts)
    mcfg.runtime_opts = getattr(mcfg, "runtime_opts", {}) or {}
    if args.workers is not None:
        mcfg.runtime_opts["workers"] = int(args.workers)
    if args.heads_per_band is not None:
        mcfg.runtime_opts["heads_per_band"] = int(args.heads_per_band)
    if args.streams is not None:
        mcfg.runtime_opts["streams"] = int(args.streams)

    # Memory watchdog config to runtime_opts
    mw = {}
    if args.mem_soft_pct is not None:
        mw["soft_pct"] = float(args.mem_soft_pct)
    if args.mem_hard_pct is not None:
        mw["hard_pct"] = float(args.mem_hard_pct)
    if args.mem_soft_mb is not None:
        mw["soft_mb"] = float(args.mem_soft_mb)
    if args.mem_hard_mb is not None:
        mw["hard_mb"] = float(args.mem_hard_mb)
    if args.mem_every is not None:
        mw["poll_ms"] = int(max(50, args.mem_every))
    if args.mem_action is not None:
        mw["action"] = str(args.mem_action)
    if mw:
        mcfg.runtime_opts["mem_watchdog"] = mw
    # Ray dashboard / lifecycle opts
    if args.ray_local_mode is not None:
        mcfg.runtime_opts["ray_local_mode"] = int(args.ray_local_mode)
    if args.ray_dashboard:
        mcfg.runtime_opts["ray_dashboard"] = True
        mcfg.runtime_opts["ray_dashboard_port"] = int(args.ray_dashboard_port)
    mcfg.runtime_opts["ray_keep_alive"] = bool(args.ray_keep_alive)

    print("Generating ...")
    import time
    @torch.no_grad()
    def _greedy_gen_timed(prefill_tokens: torch.Tensor, max_len: int):
        """Return generated tokens, state, prefill_time, decode_time.
        Optionally logs per-step stats to CSV if --stats-log is provided.
        """
        state = None
        B = prefill_tokens.size(0)
        gen = torch.empty((B, max_len), dtype=torch.long, device=device)

        stats_fp = None
        if args.stats_log:
            stats_path = Path(args.stats_log)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_fp = open(stats_path, "w")
            stats_fp.write("step,dt_ms,cum_ms,inst_tok_s,avg_tok_s\n")

        # Prefill step (full prompt)
        gauges = make_gauges()
        t0 = time.time()
        logits, state = model(prefill_tokens, state)
        t1 = time.time()
        # Emit prefill throughput if time is non-trivial
        prompt_len = prefill_tokens.shape[1]
        prefill_tps = prompt_len / max((t1 - t0), 1e-9)
        gauges["tok_s_prefill"].set(prefill_tps)
        next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
        gen[:, 0:1] = next_tok
        # Stop tracking
        stop_strings = args.stop_string or []
        eos_id = None if args.no_eos_stop else getattr(tokenizer, "eos_token_id", None)
        # Text buffer to detect stop strings incrementally
        text_buf = ""
        frag = tokenizer.decode(next_tok[0].tolist(), skip_special_tokens=False)
        text_buf += frag
        if eos_id is not None and int(next_tok.item()) == int(eos_id):
            if stats_fp:
                stats_fp.close()
            return gen[:, :1], state, (t1 - t0), 0.0
        if stop_strings and any((s and s in text_buf) for s in stop_strings):
            if stats_fp:
                stats_fp.close()
            return gen[:, :1], state, (t1 - t0), 0.0

        # Optional CfC calibrator state
        cfc_state = None

        # Decode steps
        decode_time = 0.0
        cum = 0.0
        for i in range(1, max_len):
            if _ABORT_FLAG["stop"]:
                break
            td0 = time.time()
            logits, state = model(next_tok, state)
            # Experimental CfC logit calibration (per-step)
            if args.cfc_calibrate != "off":
                from xlstm_torch.kernels.torch.experiments.cfc_logit_calibrator import CfCLogitCalibrator
                if not hasattr(_greedy_gen_timed, "_cfc_inst"):
                    _greedy_gen_timed._cfc_inst = CfCLogitCalibrator(
                        vocab_size=logits.size(-1),
                        hidden=args.cfc_hidden,
                        backbone_units=args.cfc_backbone,
                        backbone_layers=1,
                        mode=args.cfc_mode,
                        activation=("lecun_tanh" if args.cfc_calibrate=="lecun_tanh" else "lecun_tanh"),
                        topk_bias=max(0, int(args.cfc_topk)),
                    ).to(device)
                calibrator = _greedy_gen_timed._cfc_inst
                logits, cfc_state = calibrator(logits, cfc_state, token_ids=next_tok.squeeze(1))
            td1 = time.time()
            dt = td1 - td0
            decode_time += dt
            cum += dt
            next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
            gen[:, i:i+1] = next_tok
            # Stop on EOS or configured stop strings
            if eos_id is not None and int(next_tok.item()) == int(eos_id):
                break
            frag = tokenizer.decode(next_tok[0].tolist(), skip_special_tokens=False)
            text_buf += frag
            if stop_strings and any((s and s in text_buf) for s in stop_strings):
                break
            if stats_fp and (i % max(args.stats_every, 1) == 0):
                inst = 1.0 / max(dt, 1e-9)
                avg = i / max(cum, 1e-9)
                stats_fp.write(f"{i},{dt*1000.0:.3f},{cum*1000.0:.3f},{inst:.3f},{avg:.3f}\n")
            # Emit decode instantaneous tok/s to Ray dashboard if available
            gauges["tok_s_decode"].set(1.0 / max(dt, 1e-9))
        if stats_fp:
            stats_fp.close()
        return gen, state, (t1 - t0), decode_time

    mem_mon = MemoryMonitor(log_csv_path=args.mem_log).start() if args.mem_log else None
    try:
        tokens, _, t_prefill, t_decode = _greedy_gen_timed(inputs, args.max_new_tokens)
    finally:
        if mem_mon is not None:
            mem_mon.stop()
    dt = t_prefill + t_decode
    print("\nOutput:")
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(text)
    # Metrics: tokens/sec for decode and prefill throughput (tokens per second)
    prompt_len = inputs.shape[1]
    dec_tps = (args.max_new_tokens - 1) / max(t_decode, 1e-9)
    prefill_tps = prompt_len / max(t_prefill, 1e-9)
    print(f"\nTiming: total={dt:.2f}s (prefill={t_prefill:.2f}s, decode={t_decode:.2f}s)")
    print(f"Throughput: prefill={prefill_tps:.1f} tok/s, decode={dec_tps:.1f} tok/s")


    # Install lightweight signal handlers for operational control
    def _sigusr1_handler(signum, frame):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # type: ignore[attr-defined]
            print("[runner] SIGUSR1: torch.mps.empty_cache(): ok", flush=True)

    def _sigterm_handler(signum, frame):
        _ABORT_FLAG["stop"] = True
        print("[runner] SIGTERM: graceful stop requested", flush=True)

    signal.signal(signal.SIGUSR1, _sigusr1_handler)  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, _sigterm_handler)
if __name__ == "__main__":
    main()
