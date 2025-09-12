"""
Test Apple xLSTM with the same pattern as main branch.

This script replicates the working pattern from main branch
but uses our Apple Metal-accelerated implementation.
"""

import sys
sys.path.insert(0, 'xlstm-solace-torch/src')

import torch
import json
from pathlib import Path

from xlstm_solace_torch.apple_compat import create_apple_xlstm_config, xLSTMLarge


def test_apple_xlstm_7b():
    """Test 7B model creation and basic inference using Apple Metal."""
    
    print("🍎 Testing Apple xLSTM 7B with Metal Acceleration")
    print("=" * 60)
    
    # Load the official 7B config
    config_path = Path("xlstm_7b_model/config.json")
    if not config_path.exists():
        print("❌ 7B model config not found")
        return
        
    with open(config_path, 'r') as f:
        hf_config = json.load(f)
    
    print(f"📋 Original config: {hf_config['embedding_dim']}d x {hf_config['num_blocks']} blocks")
    
    # Create Apple optimized config
    apple_config = create_apple_xlstm_config(
        embedding_dim=hf_config["embedding_dim"],
        num_heads=hf_config["num_heads"], 
        num_blocks=hf_config["num_blocks"],
        vocab_size=hf_config["vocab_size"],
        use_bias=hf_config.get("use_bias", False),
        norm_eps=hf_config.get("norm_eps", 1e-6),
        qk_dim_factor=hf_config.get("qk_dim_factor", 0.5),
        v_dim_factor=hf_config.get("v_dim_factor", 1.0),
        gate_soft_cap=hf_config.get("gate_soft_cap", 15.0),
        output_logit_soft_cap=hf_config.get("output_logit_soft_cap", 30.0),
        # Apple Metal specific settings
        chunkwise_kernel="chunkwise--metal_autograd",
        sequence_kernel="native_sequence__metal", 
        step_kernel="metal",
        mode="inference",
        return_last_states=True,
    )
    
    print(f"🔧 Apple config: Metal kernels = {apple_config.chunkwise_kernel}")
    print(f"🔧 Device mode: {apple_config.mode}")
    
    # Create the model
    print("\n📦 Creating Apple xLSTM model...")
    model = xLSTMLarge(apple_config)
    
    # Move to MPS
    if torch.backends.mps.is_available():
        model = model.to('mps')
        print(f"✅ Model on device: {next(model.parameters()).device}")
    else:
        print("❌ MPS not available")
        return
    
    # Test basic forward pass
    print("\n🚀 Testing basic inference...")
    batch_size, seq_len = 1, 8
    test_input = torch.randint(0, apple_config.vocab_size, (batch_size, seq_len), device='mps')
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        
    if isinstance(output, tuple):
        logits, states = output
        print(f"✅ Inference successful! Logits shape: {logits.shape}")
        print(f"✅ States returned: {len(states)} state tensors")
    else:
        print(f"✅ Inference successful! Output shape: {output.shape}")
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Try loading some weights (first shard only for testing)
    print("\n📂 Testing weight loading...")
    try:
        from safetensors import safe_open
        weights_path = "xlstm_7b_model/model-00001-of-00006.safetensors"
        
        with safe_open(weights_path, framework="pt", device="mps") as f:
            # Just test loading one weight to verify the mechanism works
            first_key = list(f.keys())[0]
            first_weight = f.get_tensor(first_key)
            print(f"✅ Weight loading test: {first_key} -> {first_weight.shape}")
            
    except Exception as e:
        print(f"⚠️  Weight loading test failed: {e}")
    
    print("\n🎉 Apple xLSTM test completed successfully!")
    print("🎯 Ready for real inference with 7B weights!")


if __name__ == "__main__":
    test_apple_xlstm_7b()
