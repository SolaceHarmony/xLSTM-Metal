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

mlx_params = dict(model.named_parameters())
print(f"MLX model has {len(mlx_params)} parameters:")
for name in sorted(mlx_params.keys())[:20]:
    print(f"  {name}")
print(f"... and {len(mlx_params)-20} more parameters\n")

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
for name in mlx_params.keys():
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
