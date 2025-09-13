import argparse
import os
from typing import Optional, Any, Dict
from pathlib import Path

import mlx.core as mx

from .api import create_xlstm_model
from .tools.mlx_runtime import configure_gemm, configure_qr

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes([int(x) for x in ids]).decode("utf-8", errors="ignore")

def _load_hf_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def softmax_sample(logits: mx.array, temperature: float = 1.0, top_k: int = 0) -> int:
    x = logits.astype(mx.float32)
    if temperature != 1.0:
        x = x / max(1e-6, float(temperature))
    if top_k and top_k > 0 and top_k < int(x.shape[-1]):
        kth = mx.topk(x, k=top_k, axis=-1)
        thresh = kth.values[..., -1]
        mask = x < mx.expand_dims(thresh, -1)
        x = mx.where(mask, -1e30, x)
    x = x - mx.max(x, axis=-1, keepdims=True)
    p = mx.exp(x)
    p = p / mx.sum(p, axis=-1, keepdims=True)
    u = mx.random.uniform(shape=p.shape, low=0.0, high=1.0, dtype=p.dtype)
    c = mx.cumsum(p, axis=-1)
    idx = mx.argmax(u <= c, axis=-1)
    return int(idx)

def main(argv: Optional[list[str]] = None) -> None:
    print("=== MLX CLI Starting ===")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="Hello, world!")
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)

    # GGUF (optional, uses mlx_lm if provided)
    ap.add_argument("--repo", type=str, default=None, help="Hugging Face repo for GGUF model (e.g., TheBloke/Mistral-7B-v0.1-GGUF)")
    ap.add_argument("--gguf", type=str, default=None, help="GGUF filename inside the repo or local path (e.g., mistral-7b-v0.1.Q8_0.gguf)")

    # Model dims (custom Solace MLX model path)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--model-dim", type=int, default=512)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--signature", type=str, default="1,1", help="mLSTM,sLSTM pattern (e.g., 1,1)")
    ap.add_argument("--dropout", type=float, default=0.0)
    # Optional .safetensors weights for MLX module
    ap.add_argument("--weights", type=str, default=None, help="Path to .safetensors weights matching this MLX model")
    ap.add_argument("--strict", type=int, default=1, choices=[0,1], help="1 = strict (default), 0 = non-strict load")

    # Tokenizer
    ap.add_argument("--hf-tokenizer", type=str, default=None, help="Optional HF tokenizer name (e.g., gpt2)")
    # Runtime config (no envs)
    ap.add_argument("--gemm-pad", type=int, default=None)
    ap.add_argument("--gemm-align-execw", type=int, default=None)
    ap.add_argument("--gemm-double-buffer", type=int, default=None)
    ap.add_argument("--qr-dot-mode", type=str, default=None, choices=["auto","simd","simple"]) 
    # Benchmarking / stats
    ap.add_argument("--stats-log", type=str, default=None, help="Optional CSV file to log per-step decode timing (step,elapsed_s,tok_s)")
    ap.add_argument("--stats-every", type=int, default=1, help="Log every N decode steps (default 1)")
    ap.add_argument("--print-config", action="store_true", help="Print effective model/runtime config and exit")
    ap.add_argument("--profile", type=str, default=None, help="Profile name under ./configs (e.g., mlx_hardware_params)")
    ap.add_argument("--config", type=str, default=None, help="Optional JSON file with runtime overrides")
    args = ap.parse_args(argv)
    print("=== Arguments parsed ===")

    # Import runtime config functions
    from .tools.mlx_runtime import configure_gemm, configure_qr, configure_model
    print("=== Runtime tools imported ===")

    # (GGUF handling moved after JSON overrides)

    # Apply runtime config (Solace MLX custom model path)
    configure_gemm(pad=bool(args.gemm_pad) if args.gemm_pad is not None else None,
                   align_execw=bool(args.gemm_align_execw) if args.gemm_align_execw is not None else None,
                   double_buffer=bool(args.gemm_double_buffer) if args.gemm_double_buffer is not None else None)
    if args.qr_dot_mode is not None:
        configure_qr(dot_mode=args.qr_dot_mode)

    # Resolve prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

    # Tokenizer
    tok = None
    if args.hf_tokenizer:
        tok = _load_hf_tokenizer(args.hf_tokenizer)
        vocab_size = int(tok.vocab_size)
    else:
        tok = ByteTokenizer()
        vocab_size = int(args.vocab_size)

    # JSON config overlays (defaults → profile → --config)
    def _load_json(fp):
        import json as _json
        from pathlib import Path as _P
        p = _P(fp)
        if not p.exists():
            return {}
        return _json.loads(p.read_text())
    base = _load_json("configs/mlx_hardware_params.json")
    prof = {}
    # Load profile from local configs if provided
    if args.profile:
        from pathlib import Path as _P
        prof = _load_json(str((_P("configs") / f"{args.profile}.json")))
    # Optional explicit config file
    cfg_file = _load_json(args.config) if args.config else {}
    merged = {}
    # Always try packaged golden to provide polished defaults unless overridden later
    import importlib.resources as ir
    with ir.files('xlstm_mlx.configs').joinpath('mlx_golden.json').open('r') as f:
        import json as _json
        merged.update(_json.load(f))
    for d in (base, prof, cfg_file):
        merged.update({k: v for k, v in d.items() if v is not None})
    # Allow model and run-level options in config JSON as well
    def _apply_json_overrides(args_obj, cfg: Dict[str, Any]):
        mapping = {
            # Run options
            "prompt": "prompt",
            "prompt_file": "prompt_file",
            "max_new_tokens": "max_new_tokens",
            "temperature": "temperature",
            "top_k": "top_k",
            "stats_log": "stats_log",
            "stats_every": "stats_every",
            "hf_tokenizer": "hf_tokenizer",
            # Weights / strict
            "weights": "weights",
            "strict": "strict",
            # Model dims
            "vocab_size": "vocab_size",
            "layers": "layers",
            "model_dim": "model_dim",
            "head_dim": "head_dim",
            "heads": "heads",
            "signature": "signature",
            # Runtime tuning
            "gemm_pad": "gemm_pad",
            "gemm_align_execw": "gemm_align_execw",
            "gemm_double_buffer": "gemm_double_buffer",
            "qr_dot_mode": "qr_dot_mode",
            "fast_head": "fast_head",
            # GGUF via mlx_lm
            "repo": "repo",
            "gguf": "gguf",
            # Conversion helpers (HF → MLX)
            "hf_path": "hf_path",
            "mlx_out": "mlx_out",
            "convert_hf": "convert_hf",
        }
        for k_src, k_dst in mapping.items():
            if k_src in cfg:
                setattr(args_obj, k_dst, cfg[k_src])
        # Normalize types
        if isinstance(getattr(args_obj, "strict", 1), int):
            setattr(args_obj, "strict", int(getattr(args_obj, "strict")))
        return args_obj
    args = _apply_json_overrides(args, merged)

    # If GGUF is specified (possibly via JSON), use mlx_lm to load and run
    if getattr(args, "gguf", None) and getattr(args, "repo", None):
        try:
            from mlx_lm import load as _mlx_load, generate as _mlx_generate  # type: ignore
        except Exception as e:
            raise RuntimeError("mlx_lm is required for GGUF loading. Install with: pip install mlx-lm") from e
        model, tokenizer = _mlx_load(args.repo, tokenizer_config={}, model=args.gguf)
        text = _mlx_generate(model, tokenizer, prompt=args.prompt, max_tokens=int(args.max_new_tokens))
        print(text)
        return
    # Optional HF → MLX conversion if JSON provides hf_path and no explicit weights
    if getattr(args, "weights", None) in (None, "") and merged.get("hf_path") and bool(merged.get("convert_hf", False)):
        try:
            from mlx_lm.convert import convert  # type: ignore
        except Exception as e:
            raise RuntimeError("mlx_lm is required for HF→MLX conversion. Install with: pip install mlx-lm") from e
        hf_path = Path(str(merged["hf_path"]))
        mlx_out = Path(str(merged.get("mlx_out", Path("model_cache") / (hf_path.name + "_mlx"))))
        # If target exists and has weights, reuse; if exists but empty, create a fresh subdir
        if mlx_out.exists():
            st = sorted(mlx_out.glob("*.safetensors"))
            if not st:
                # Find a unique subdir to satisfy mlx_lm.convert requirement
                base = mlx_out
                i = 1
                while True:
                    candidate = base.parent / f"{base.name}_{i}"
                    if not candidate.exists():
                        mlx_out = candidate
                        break
                    i += 1
        # Perform conversion only if no .safetensors present at (possibly updated) mlx_out
        if not mlx_out.exists():
            # convert will create the directory; do not pre-create
            convert(str(hf_path), str(mlx_out))
        st = sorted(mlx_out.glob("*.safetensors"))
        if not st:
            raise RuntimeError(f"No .safetensors produced in {mlx_out} after conversion")
        args.weights = str(st[0])

    # Map merged into runtime config
    if "gemm_pad" in merged or "gemm_align_execw" in merged or "gemm_double_buffer" in merged:
        configure_gemm(
            pad=merged.get("gemm_pad"),
            align_execw=merged.get("gemm_align_execw"),
            double_buffer=merged.get("gemm_double_buffer"),
        )
    if "qr_dot_mode" in merged:
        configure_qr(dot_mode=merged.get("qr_dot_mode"))
    if "fast_head" in merged:
        configure_model(fast_head=merged.get("fast_head"))

    # Model
    # Signature can be provided as string "1,1" or list
    if isinstance(args.signature, str):
        sig = tuple(int(x) for x in args.signature.split(","))
    else:
        sig = tuple(int(x) for x in (args.signature or [1, 1]))
    if args.print_config:
        eff = {
            "vocab_size": vocab_size,
            "layers": int(args.layers),
            "signature": sig,
            "model_dim": int(args.model_dim),
            "head_dim": int(args.head_dim),
            "heads": int(args.heads),
            "runtime": merged
        }
        import json as _json
        print(_json.dumps(eff, indent=2))
        return
    model = create_xlstm_model(
        vocab_size=int(getattr(args, "vocab_size", vocab_size)),
        num_layers=int(args.layers),
        signature=sig,
        inp_dim=int(args.model_dim),
        head_dim=int(args.head_dim),
        head_num=int(args.heads),
        dropout=float(args.dropout),
    )

def map_hf_to_mlx_params(hf_weights):
    """Map HuggingFace parameter names to MLX format."""
    mlx_params = {}
    
    # Handle embedding layer
    if "backbone.embeddings.weight" in hf_weights:
        mlx_params["embedding.weight"] = hf_weights["backbone.embeddings.weight"]
    
    # Handle output head
    head_keys = ["lm_head.weight", "head.weight", "output.weight"]
    for key in head_keys:
        if key in hf_weights:
            mlx_params["head.W"] = hf_weights[key]
            break
    
    # Map each block
    block_keys = [k for k in hf_weights.keys() if k.startswith("backbone.blocks.")]
    blocks = set()
    for key in block_keys:
        parts = key.split(".")
        if len(parts) >= 3:
            blocks.add(int(parts[2]))
    
    for i in sorted(blocks):
        hf_prefix = f"backbone.blocks.{i}"
        mlx_prefix = f"blocks.{i}"
        
        # mLSTM layer mappings
        mlstm_mappings = {
            f"{hf_prefix}.mlstm_layer.q.weight": f"{mlx_prefix}.W_q.weight",
            f"{hf_prefix}.mlstm_layer.k.weight": f"{mlx_prefix}.W_k.weight", 
            f"{hf_prefix}.mlstm_layer.v.weight": f"{mlx_prefix}.W_v.weight",
            f"{hf_prefix}.mlstm_layer.out_proj.weight": f"{mlx_prefix}.W_o.weight",
            f"{hf_prefix}.mlstm_layer.igate_preact.weight": f"{mlx_prefix}.W_i.weight",
            f"{hf_prefix}.mlstm_layer.igate_preact.bias": f"{mlx_prefix}.W_i.bias",
            f"{hf_prefix}.mlstm_layer.fgate_preact.weight": f"{mlx_prefix}.W_f.weight",
            f"{hf_prefix}.mlstm_layer.fgate_preact.bias": f"{mlx_prefix}.W_f.bias",
            f"{hf_prefix}.norm_mlstm.weight": f"{mlx_prefix}.norm.weight",
            f"{hf_prefix}.norm_ffn.weight": f"{mlx_prefix}.norm2.weight",
            f"{hf_prefix}.mlstm_layer.multihead_norm.weight": f"{mlx_prefix}.mhln.weight",
        }
        
        for hf_key, mlx_key in mlstm_mappings.items():
            if hf_key in hf_weights:
                mlx_params[mlx_key] = hf_weights[hf_key]
        
        # FFN mappings  
        ffn_mappings = {
            f"{hf_prefix}.ffn.proj_up.weight": f"{mlx_prefix}.up_l_proj.weight",
            f"{hf_prefix}.ffn.proj_up_gate.weight": f"{mlx_prefix}.up_r_proj.weight", 
            f"{hf_prefix}.ffn.proj_down.weight": f"{mlx_prefix}.down_proj.weight",
        }
        
        for hf_key, mlx_key in ffn_mappings.items():
            if hf_key in hf_weights:
                mlx_params[mlx_key] = hf_weights[hf_key]
    
    return mlx_params

def load_hf_sharded_weights(model_path):
    """Load all sharded HuggingFace weights into a single dict."""
    import json
    from pathlib import Path
    from safetensors import safe_open
    
    model_path = Path(model_path)
    
    # Load the index to find all shard files
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        # Try single file
        single_file = model_path / "model.safetensors"
        if single_file.exists():
            weights = {}
            with safe_open(str(single_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    weights[key] = mx.array(tensor.numpy())
            return weights
        else:
            raise FileNotFoundError(f"Neither index file nor single model file found in {model_path}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    # Get all unique shard files
    shard_files = set(index["weight_map"].values())
    
    # Load all weights
    weights = {}
    for shard_file in shard_files:
        shard_path = model_path / shard_file
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = mx.array(tensor.numpy())
    
    return weights

    # Optionally load .safetensors weights into the MLX model
    if args.weights:
        wpath = Path(str(args.weights))
        if not wpath.exists():
            raise FileNotFoundError(f"Weights not found: {wpath}. Set 'weights' in your JSON to an existing MLX .safetensors file.")
        try:
            use_mx_loader = bool(merged.get("weights_loader") == "mlx")
            use_hf_mapping = bool(merged.get("hf_mapping", True))  # Enable by default
            
            if use_mx_loader:
                data = mx.load(str(wpath))  # may be a dict or single array
                if isinstance(data, dict):
                    pairs = [(k, v) for k, v in data.items()]
                else:
                    # Single-array files must map to a known single parameter name
                    raise RuntimeError("Single-array weight files are not supported without parameter names.")
                model.load_weights(pairs, strict=bool(int(args.strict)))
            elif use_hf_mapping and str(wpath).endswith('.safetensors'):
                # Try to load as HuggingFace format and map parameters
                if wpath.is_file():
                    # Single safetensors file
                    from safetensors import safe_open
                    hf_weights = {}
                    with safe_open(str(wpath), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            hf_weights[key] = mx.array(tensor.numpy())
                else:
                    # Try to load sharded weights from directory
                    wpath_dir = wpath.parent
                    hf_weights = load_hf_sharded_weights(wpath_dir)
                
                # Map HF parameters to MLX format
                mapped_params = map_hf_to_mlx_params(hf_weights)
                pairs = [(k, v) for k, v in mapped_params.items()]
                model.load_weights(pairs, strict=bool(int(args.strict)))
                print(f"Loaded and mapped {len(pairs)} parameters from HF format")
            else:
                model.load_weights(str(wpath), strict=bool(int(args.strict)))
            print(f"Loaded weights from: {args.weights} (strict={bool(int(args.strict))})")
        except Exception as e:
            raise RuntimeError(
                "Failed to load weights. Ensure the .safetensors file was saved for this MLX architecture and naming.\n"
                "For raw files, set 'weights_loader': 'mlx' in your JSON to load via mx.core.load.\n"
                "For HF format, set 'hf_mapping': true in your JSON (default)."
            ) from e

    # Encode prompt
    if args.hf_tokenizer:
        ids = tok.encode(prompt, return_tensors=None)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    else:
        ids = tok.encode(prompt)
    # Allow JSON to override max sequence at prefill by limiting ids
    ids = ids[:4096]
    input_ids = mx.array([ids], dtype=mx.int32)

    use_streams = True
    s_gpu = mx.new_stream(mx.gpu)

    # Prefill
    import time, csv
    print(f"Starting prefill with input_ids shape: {input_ids.shape}")
    t_prefill_start = time.time()
    if use_streams:
        with mx.stream(s_gpu):
            logits, state = model(input_ids)
            last_logits = logits[:, -1, :]
    else:
        logits, state = model(input_ids)
        last_logits = logits[:, -1, :]
    print(f"Prefill completed, logits shape: {logits.shape}")
    t_prefill = time.time() - t_prefill_start

    # Decode
    out_ids = list(ids)
    print(f"Starting decode loop for {int(args.max_new_tokens)} tokens")
    stats_path = args.stats_log
    stats_every = max(1, int(args.stats_every))
    csv_f = None; csv_w = None
    if stats_path:
        csv_f = open(stats_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["step", "elapsed_s", "tok_s"])

    t_decode_start = time.time()
    for step in range(1, int(args.max_new_tokens) + 1):
        print(f"Decode step {step}")
        t_step_start = time.time()
        if use_streams:
            with mx.stream(s_gpu):
                next_id = softmax_sample(last_logits[0], args.temperature, args.top_k)
                out_ids.append(int(next_id))
                step_in = mx.array([[int(next_id)]], dtype=mx.int32)
                logits, state = model(step_in, state)
                last_logits = logits[:, -1, :]
        else:
            next_id = softmax_sample(last_logits[0], args.temperature, args.top_k)
            out_ids.append(int(next_id))
            step_in = mx.array([[int(next_id)]], dtype=mx.int32)
            logits, state = model(step_in, state)
            last_logits = logits[:, -1, :]
        print(f"Generated token: {int(next_id)}")
        if (step % stats_every) == 0 and (csv_w is not None):
            elapsed = time.time() - t_step_start
            tok_s = 1.0 / max(1e-9, elapsed)
            csv_w.writerow([step, f"{elapsed:.6f}", f"{tok_s:.2f}"])

    if use_streams:
        mx.synchronize(s_gpu)

    t_decode = time.time() - t_decode_start
    total_new = int(args.max_new_tokens)
    overall_tok_s = (total_new / max(1e-9, t_decode)) if total_new > 0 else 0.0
    if csv_f is not None:
        csv_f.flush(); csv_f.close()

    if args.hf_tokenizer:
        text = tok.decode(out_ids)
    else:
        text = tok.decode(out_ids)
    print(text)
    print("\n--- Benchmark Summary (MLX) ---")
    print(f"Prefill time: {t_prefill*1000:.1f} ms")
    print(f"Decode tokens: {total_new}")
    print(f"Decode time: {t_decode:.3f} s")
    print(f"Decode throughput: {overall_tok_s:.1f} tok/s")

if __name__ == "__main__":
    os.environ.setdefault("XLSTM_MLX_FAST_HEAD", "1")
    main()
