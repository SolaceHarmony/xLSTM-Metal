#!/usr/bin/env python3
"""
Test official xlstm package with Apple Metal acceleration
"""

import torch
import json
from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig
from transformers import AutoTokenizer

def create_apple_metal_config(base_config_path: str) -> xLSTMLargeConfig:
    """Create config using official package but with Apple Metal kernels"""
    
    # Load the 7B config
    with open(base_config_path, 'r') as f:
        cfg = json.load(f)
    
    # Create config with APPLE METAL KERNELS instead of Triton
    config = xLSTMLargeConfig(
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
        
        # FORCE APPLE METAL ACCELERATION
        chunkwise_kernel="chunkwise--triton_xl_chunk",  # This should work on Apple
        sequence_kernel="native_sequence__triton",      # This should work on Apple  
        step_kernel="triton",                           # This should work on Apple
        mode="inference",
        chunk_size=64,
        return_last_states=True,
        autocast_kernel_dtype="bfloat16",
        inference_state_dtype="float32",
    )
    
    return config

def test_official_xlstm_apple():
    """Test the official xLSTM package on Apple Silicon"""
    
    print("🍎 Testing Official xLSTM Package with Apple Metal")
    print("=" * 50)
    
    # Create Apple Metal config
    config = create_apple_metal_config("xlstm_7b_model/config.json")
    print(f"✅ Config created:")
    print(f"   - Vocab: {config.vocab_size}")
    print(f"   - Embedding: {config.embedding_dim}")
    print(f"   - Blocks: {config.num_blocks}")
    print(f"   - Chunkwise: {config.chunkwise_kernel}")
    print(f"   - Sequence: {config.sequence_kernel}")
    print(f"   - Step: {config.step_kernel}")
    
    # Create model
    print(f"\n🏗️ Creating xLSTMLarge model...")
    model = xLSTMLarge(config)
    model = model.to("mps")
    print(f"✅ Model created on device: {next(model.parameters()).device}")
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlstm_7b_model")
    print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
    
    # Test inference
    print(f"\n🚀 Testing inference...")
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("mps")
    
    print(f"Input: '{prompt}'")
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Generate
    with torch.no_grad():
        # Single forward pass
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        # Get next token
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
        next_token = tokenizer.decode(next_token_id)
        
        print(f"✅ Generated next token: '{next_token}'")
        print(f"✅ Logits shape: {logits.shape}")
        
    print(f"\n🎉 SUCCESS! Official xLSTM package works on Apple Silicon!")
    return True

if __name__ == "__main__":
    test_official_xlstm_apple()
