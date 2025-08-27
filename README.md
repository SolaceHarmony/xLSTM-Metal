# xLSTM Implementation Suite

A comprehensive collection of xLSTM (Extended LSTM) implementations with varying optimization levels, from basic mathematical prototypes to production-ready systems.

## Overview

This repository contains multiple xLSTM implementations progressing from research prototypes to optimized production systems. The implementations maintain mathematical fidelity to the original xLSTM paper while adding practical enhancements for real-world usage.

## Implementation Status

### Core Implementations

| Implementation | Status | Performance | Features |
|----------------|--------|-------------|----------|
| `xlstm_pytorch_enhanced.py` | ✅ Stable | 730 tok/s | Production-ready, comprehensive config |
| `xlstm_pytorch.py` | ✅ Stable | ~400 tok/s | Base implementation |
| `xlstm_mlx.py` | ✅ Stable | ~300 tok/s | Apple Silicon MLX backend |
| `xlstm_metal_optimized.py` | ⚠️ Beta | TBD | MPS optimization, torch.compile |
| `xlstm_chunked_parallel.py` | ⚠️ Beta | TBD | Chunk-based parallelization |
| `xlstm_streaming_inference.py` | ⚠️ Beta | TBD | Real-time streaming |

### Testing and Utilities

- `xlstm_ultimate_benchmark.py` - Comprehensive benchmarking suite
- `run_tests.py` - Test suite for all implementations
- `train_xlstm.py` - Training framework
- `benchmark.py` - Performance testing utilities

## Key Features

### sLSTM Block
- Exponential gating mechanism preventing gradient vanishing
- State normalization for numerical stability
- Memory mixing within heads
- Optional causal convolution

### mLSTM Block
- Matrix memory structure for enhanced capacity
- Covariance update rule
- Fully parallelizable operations
- Skip connections and grouped normalization

## Installation

```bash
pip install -r requirements.txt
```

For MLX (Apple Silicon only):
```bash
pip install mlx
```

## Usage

### MLX Version
```python
from xlstm_mlx import create_xlstm_model

model = create_xlstm_model(
    vocab_size=50257,
    num_layers=12,
    signature=(7, 1),  # 7 mLSTM blocks, 1 sLSTM block pattern
    inp_dim=768,
    head_dim=96,
    head_num=8
)

# Forward pass
tokens = mx.random.randint(0, 50257, (batch_size, seq_len))
logits = model(tokens)
```

### PyTorch Version
```python
from xlstm_pytorch import create_xlstm_model

model = create_xlstm_model(
    vocab_size=50257,
    num_layers=12,
    signature=(7, 1),  # 7 mLSTM blocks, 1 sLSTM block pattern
    inp_dim=768,
    head_dim=96,
    head_num=8,
    device='cuda'
)

# Forward pass
tokens = torch.randint(0, 50257, (batch_size, seq_len))
logits = model(tokens)
```

## Testing

Run the test suite to validate both implementations:

```bash
python test_implementations.py
```

## Model Configuration

Key parameters:
- `vocab_size`: Size of the vocabulary
- `num_layers`: Total number of xLSTM blocks
- `signature`: Tuple (num_mLSTM, num_sLSTM) defining the block pattern
- `inp_dim`: Input/embedding dimension
- `head_dim`: Dimension per attention head
- `head_num`: Number of attention heads
- `p_factor`: Projection factors for (mLSTM, sLSTM)
- `ker_size`: Kernel size for causal convolution
- `dropout`: Dropout probability

## Architecture Details

The xLSTM architecture cycles through the specified signature pattern. For example, with `signature=(7, 1)` and `num_layers=16`, the model will have:
- Blocks 0-6: mLSTM
- Block 7: sLSTM
- Blocks 8-14: mLSTM
- Block 15: sLSTM

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [Official xLSTM Repository](https://github.com/NX-AI/xlstm)
- [MLX xLSTM Implementation](https://github.com/abeleinin/mlx-xLSTM)

## License

This implementation is provided for educational and research purposes.