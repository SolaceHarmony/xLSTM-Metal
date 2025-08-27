#!/usr/bin/env python
"""
Test script to validate xLSTM implementations for both MLX and PyTorch
"""

import sys
import numpy as np

def test_mlx_implementation():
    """Test MLX xLSTM implementation"""
    print("Testing MLX implementation...")
    try:
        import mlx.core as mx
        from xlstm_mlx import create_xlstm_model
        
        # Create a small model for testing
        model = create_xlstm_model(
            vocab_size=100,
            num_layers=2,
            signature=(1, 1),
            inp_dim=64,
            head_dim=16,
            head_num=4,
            dropout=0.0
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        tokens = mx.random.randint(0, 100, (batch_size, seq_len))
        
        logits = model(tokens)
        print(f"  MLX Output shape: {logits.shape}")
        print(f"  MLX Output dtype: {logits.dtype}")
        
        # Test with hidden states
        hidden_states = model.init_hidden(batch_size)
        logits_with_hidden, new_hidden = model(tokens, hidden_states, return_hidden=True)
        print(f"  MLX with hidden states output shape: {logits_with_hidden.shape}")
        print(f"  MLX Number of hidden states: {len(new_hidden)}")
        
        print("  MLX implementation test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"  MLX implementation test FAILED: {e}")
        return False


def test_pytorch_implementation():
    """Test PyTorch xLSTM implementation"""
    print("\nTesting PyTorch implementation...")
    try:
        import torch
        from xlstm_pytorch import create_xlstm_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")
        
        # Create a small model for testing
        model = create_xlstm_model(
            vocab_size=100,
            num_layers=2,
            signature=(1, 1),
            inp_dim=64,
            head_dim=16,
            head_num=4,
            dropout=0.0,
            device=device
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        tokens = torch.randint(0, 100, (batch_size, seq_len), device=device)
        
        logits = model(tokens)
        print(f"  PyTorch Output shape: {logits.shape}")
        print(f"  PyTorch Output dtype: {logits.dtype}")
        
        # Test with hidden states
        hidden_states = model.init_hidden(batch_size)
        logits_with_hidden, new_hidden = model(tokens, hidden_states, return_hidden=True)
        print(f"  PyTorch with hidden states output shape: {logits_with_hidden.shape}")
        print(f"  PyTorch Number of hidden states: {len(new_hidden)}")
        
        # Test gradient computation
        loss = logits.mean()
        loss.backward()
        has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  PyTorch Gradients computed: {has_gradients}")
        
        print("  PyTorch implementation test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"  PyTorch implementation test FAILED: {e}")
        return False


def test_consistency():
    """Test that both implementations produce outputs with same shapes"""
    print("\nTesting consistency between implementations...")
    
    try:
        # Import both implementations
        import mlx.core as mx
        import torch
        from xlstm_mlx import create_xlstm_model as create_mlx_model
        from xlstm_pytorch import create_xlstm_model as create_pytorch_model
        
        # Use same configuration for both
        config = {
            'vocab_size': 50,
            'num_layers': 2,
            'signature': (1, 1),
            'inp_dim': 32,
            'head_dim': 8,
            'head_num': 4,
            'dropout': 0.0
        }
        
        # Create models
        mlx_model = create_mlx_model(**config)
        
        device = 'cpu'  # Use CPU for fair comparison
        pytorch_model = create_pytorch_model(**config, device=device)
        
        # Test with same input size
        batch_size = 2
        seq_len = 5
        
        # MLX forward pass
        mlx_tokens = mx.random.randint(0, config['vocab_size'], (batch_size, seq_len))
        mlx_logits = mlx_model(mlx_tokens)
        
        # PyTorch forward pass
        pytorch_tokens = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        pytorch_logits = pytorch_model(pytorch_tokens)
        
        # Compare shapes
        mlx_shape = mlx_logits.shape
        pytorch_shape = tuple(pytorch_logits.shape)
        
        print(f"  MLX output shape: {mlx_shape}")
        print(f"  PyTorch output shape: {pytorch_shape}")
        
        shapes_match = mlx_shape == pytorch_shape
        print(f"  Shapes match: {shapes_match}")
        
        if shapes_match:
            print("  Consistency test PASSED ✓")
            return True
        else:
            print("  Consistency test FAILED")
            return False
            
    except Exception as e:
        print(f"  Consistency test FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("xLSTM Implementation Tests")
    print("="*60)
    
    results = []
    
    # Test MLX implementation
    mlx_result = test_mlx_implementation()
    results.append(("MLX Implementation", mlx_result))
    
    # Test PyTorch implementation
    pytorch_result = test_pytorch_implementation()
    results.append(("PyTorch Implementation", pytorch_result))
    
    # Test consistency
    consistency_result = test_consistency()
    results.append(("Consistency Check", consistency_result))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests PASSED! ✓")
        print("Both xLSTM implementations are working correctly.")
    else:
        print("Some tests FAILED. Please review the output above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())