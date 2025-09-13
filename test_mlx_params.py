import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
mlx_src = os.path.join(root, "xlstm-solace-mlx", "src")
if mlx_src not in sys.path:
    sys.path.insert(0, mlx_src)

from xlstm_mlx.api import create_xlstm_model

# Create model with same dims as config
model = create_xlstm_model(
    vocab_size=50304,
    num_layers=32,
    signature=(1, 1),
    inp_dim=4096,
    head_dim=512,
    head_num=8
)

print("MLX Model Parameters:")
import mlx.nn as nn

def print_module_params(module, prefix=""):
    for key, value in module.__dict__.items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if hasattr(item, '__dict__'):
                    print_module_params(item, f"{prefix}{key}.{i}.")
        elif hasattr(value, 'shape'):  # It's a parameter
            print(f"  {prefix}{key}: {value.shape}")
        elif hasattr(value, '__dict__') and not key.startswith('_'):
            print_module_params(value, f"{prefix}{key}.")

print_module_params(model)
