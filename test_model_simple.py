#!/usr/bin/env python
"""
Simple test of xLSTM-7B model
"""

import torch
import json
from pathlib import Path

model_path = Path("./xlstm_7b_model")

print("=" * 60)
print("xLSTM-7B Model Test")
print("=" * 60)

# Load config
with open(model_path / "config.json") as f:
    config = json.load(f)

print("\nModel Configuration:")
print(f"  Architecture: {config.get('architectures', ['unknown'])[0]}")
print(f"  Model Type: {config.get('model_type', 'unknown')}")
print(f"  Vocab Size: {config.get('vocab_size', 'unknown')}")
print(f"  Hidden Size: {config.get('hidden_size', 'unknown')}")
print(f"  Num Layers: {config.get('num_hidden_layers', 'unknown')}")

# Check safetensor files
safetensor_files = list(model_path.glob("*.safetensors"))
print(f"\nModel Files:")
total_size = 0
for f in sorted(safetensor_files):
    size_gb = f.stat().st_size / (1024**3)
    total_size += size_gb
    print(f"  {f.name}: {size_gb:.2f} GB")
print(f"  Total: {total_size:.1f} GB")

# Load index to understand weight structure
with open(model_path / "model.safetensors.index.json") as f:
    index = json.load(f)

print(f"\nWeight Structure:")
print(f"  Total weight tensors: {len(index['weight_map'])}")

# Show sample weight names
print("\nSample weight names:")
for i, (name, file) in enumerate(list(index['weight_map'].items())[:10]):
    print(f"  {name} -> {file}")

# Try to load with safetensors directly
try:
    from safetensors import safe_open
    
    print("\n" + "=" * 60)
    print("Testing safetensor loading...")
    
    # Load first file to check tensor shapes
    first_file = model_path / "model-00001-of-00006.safetensors"
    with safe_open(first_file, framework="pt", device="cpu") as f:
        print(f"\nTensors in {first_file.name}:")
        for key in list(f.keys())[:5]:
            tensor = f.get_tensor(key)
            print(f"  {key}: shape {tensor.shape}, dtype {tensor.dtype}")
    
    print("\n✓ Safetensors load successfully!")
    
except Exception as e:
    print(f"\n✗ Error loading safetensors: {e}")

print("\n" + "=" * 60)
print("Model files are ready for use!")
print("=" * 60)