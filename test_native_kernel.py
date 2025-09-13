"""
Quick test with native kernel to verify the structure works.
"""

import sys
sys.path.insert(0, 'xlstm-solace-torch/src')

import torch
from xlstm_torch.xlstm_large import xLSTMLarge, xLSTMLargeConfig

# Create config with native kernel instead of Metal
config = xLSTMLargeConfig(
    embedding_dim=128,
    num_heads=4,
    num_blocks=2,
    vocab_size=1000,
    # Use native kernel instead of Metal
    chunkwise_kernel="chunkwise--native_autograd",
    sequence_kernel="native_sequence__autograd", 
    step_kernel="native",
)

print(f"🔧 Testing with native kernels: {config.chunkwise_kernel}")

# Create model
model = xLSTMLarge(config)
print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward - no MPS needed for native kernels
test_input = torch.randint(0, config.vocab_size, (1, 8))
output = model(test_input)

if isinstance(output, tuple):
    logits, states = output
    print(f"✅ Native kernel works: logits {logits.shape}, {len(states)} states")
else:
    print(f"✅ Native kernel works: output {output.shape}")

print("🎯 Structure is correct - issue is specifically with Metal backend loading")
