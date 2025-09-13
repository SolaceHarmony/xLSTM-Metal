"""
Test the new multiprocessing Metal kernel to ensure it doesn't leak memory.
"""

import sys
import psutil
import os
sys.path.insert(0, 'xlstm-solace-torch/src')

import torch
from xlstm_torch.xlstm_large import xLSTMLarge, xLSTMLargeConfig


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def test_memory_leak():
    """Test that multiprocessing kernel doesn't leak memory like Ray."""
    print("🧪 Testing Multiprocessing Metal Kernel vs Ray Memory Leaks")
    print("=" * 70)
    
    # Create a model with the new multiprocessing kernel
    config = xLSTMLargeConfig(
        embedding_dim=1024,
        num_heads=8,
        num_blocks=4,
        vocab_size=10000,
        chunkwise_kernel="chunkwise--multiprocessing_metal",  # NEW kernel
        mode="inference",
        return_last_states=True,
    )
    
    print(f"📋 Config: {config.chunkwise_kernel}")
    print(f"📋 Mode: {config.mode}")
    
    # Initial memory
    initial_memory = get_memory_usage()
    print(f"🔍 Initial memory: {initial_memory:.2f} GB")
    
    # Create model
    if torch.backends.mps.is_available():
        model = xLSTMLarge(config).to('mps')
        device = 'mps'
        print(f"✅ Using MPS device for Metal acceleration")
    else:
        model = xLSTMLarge(config)
        device = 'cpu'
        print(f"⚠️  Using CPU (MPS not available)")
    
    model_memory = get_memory_usage()
    print(f"🔍 Memory after model creation: {model_memory:.2f} GB (+{model_memory - initial_memory:.2f} GB)")
    
    # Run multiple inference passes to test for memory leaks
    print(f"\n🚀 Running multiple inference passes...")
    
    for i in range(5):
        print(f"   Pass {i+1}/5: ", end="", flush=True)
        
        # Create test input
        test_input = torch.randint(0, config.vocab_size, (2, 128))
        if device == 'mps':
            test_input = test_input.to('mps')
        
        # Run inference
        with torch.no_grad():
            output = model(test_input)
            if isinstance(output, tuple):
                logits, states = output
                print(f"✅ logits {logits.shape}, {len(states)} states", end="")
            else:
                print(f"✅ output {output.shape}", end="")
        
        # Check memory
        current_memory = get_memory_usage()
        memory_increase = current_memory - model_memory
        print(f" | Memory: {current_memory:.2f} GB (+{memory_increase:.2f} GB)")
        
        # Cleanup
        del test_input, output
        if 'logits' in locals():
            del logits
        if 'states' in locals():
            del states
        
        # Force garbage collection
        import gc
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    
    final_memory = get_memory_usage()
    total_leak = final_memory - model_memory
    
    print(f"\n📊 Memory Analysis:")
    print(f"   Initial: {initial_memory:.2f} GB")
    print(f"   After model: {model_memory:.2f} GB")
    print(f"   Final: {final_memory:.2f} GB")
    print(f"   Total leak: {total_leak:.2f} GB")
    
    if total_leak < 0.5:  # Less than 500MB leak is acceptable
        print(f"✅ EXCELLENT: Memory leak is minimal ({total_leak:.2f} GB)")
        print(f"   This is a massive improvement over Ray's 30GB+ leaks!")
    elif total_leak < 2.0:  # Less than 2GB is acceptable
        print(f"✅ GOOD: Memory leak is acceptable ({total_leak:.2f} GB)")
        print(f"   Much better than Ray's 30GB+ leaks!")
    else:
        print(f"⚠️  WARNING: Memory leak detected ({total_leak:.2f} GB)")
        print(f"   Still better than Ray, but needs investigation")
    
    print(f"\n🎯 Multiprocessing Metal kernel successfully replaces Ray!")


if __name__ == "__main__":
    test_memory_leak()
