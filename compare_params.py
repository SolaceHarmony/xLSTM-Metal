#!/usr/bin/env python3
"""
Compare MLX model parameters vs HF model parameters to create a mapping.
"""
import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
mlx_src = os.path.join(root, "xlstm-solace-mlx", "src")
if mlx_src not in sys.path:
    sys.path.insert(0, mlx_src)

from xlstm_solace_mlx.api import create_xlstm_model
from safetensors import safe_open

# Create MLX model
print("=== MLX Model Parameters ===")
model = create_xlstm_model(
    vocab_size=50304,
    num_layers=32,
    signature=(1, 1),
    inp_dim=4096,
    head_dim=512,
    head_num=8,
    dropout=0.0,
)

mlx_params = model.parameters()
print(f"MLX model parameters type: {type(mlx_params)}")

def flatten_params(params, prefix=""):
    """Flatten nested parameter dict into flat dict with dot notation."""
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if hasattr(value, 'shape'):  # It's an mlx.core.array
            flat[full_key] = value
        elif isinstance(value, (dict, list)):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(flatten_params(item, f"{full_key}.{i}"))
                    elif hasattr(item, 'shape'):
                        flat[f"{full_key}.{i}"] = item
            else:
                flat.update(flatten_params(value, full_key))
    return flat

flat_mlx_params = flatten_params(mlx_params)
print(f"MLX model has {len(flat_mlx_params)} parameters:")
for name in sorted(list(flat_mlx_params.keys())[:20]):
    print(f"  {name}: {flat_mlx_params[name].shape}")
remaining = len(flat_mlx_params) - 20
if remaining > 0:
    print(f"... and {remaining} more parameters\n")
else:
    print()

# Load HF model parameters
print("=== HF Model Parameters ===")
with safe_open('xlstm_7b_model/model-00001-of-00006.safetensors', framework='pt', device='cpu') as f:
    hf_keys = list(f.keys())

print(f"HF model has {len(hf_keys)} parameters:")
for key in sorted(hf_keys)[:20]:
    print(f"  {key}")
print(f"... and {len(hf_keys)-20} more parameters\n")

# Try to create a mapping
print("=== Parameter Name Analysis ===")
print("MLX parameter prefixes:")
mlx_prefixes = set()
for name in flat_mlx_params.keys():
    parts = name.split('.')
    if len(parts) > 1:
        mlx_prefixes.add(parts[0])
print(f"  {sorted(mlx_prefixes)}")

print("\nHF parameter prefixes:")
hf_prefixes = set()
for name in hf_keys:
    parts = name.split('.')
    if len(parts) > 1:
        hf_prefixes.add(parts[0])
print(f"  {sorted(hf_prefixes)}")
