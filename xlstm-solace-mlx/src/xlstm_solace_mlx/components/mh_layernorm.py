import mlx.core as mx
import mlx.nn as nn

class MultiHeadLayerNormMLX(nn.Module):
    """Multi‑Head LayerNorm in MLX (per head, over DH).

    Normalizes the last dimension per head for concatenated head features.
    Expects input x of shape (B, H*DH). Applies per‑head epsilon, and optional
    learned scale/bias of shape (H, DH), then flattens back to (B, H*DH).
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6,
                 use_weight: bool = True, use_bias: bool = False):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        H, DH = self.num_heads, self.head_dim
        # MLX registers arrays assigned to attributes as parameters
        self.weight = mx.ones((H, DH)) if use_weight else None
        self.bias = mx.zeros((H, DH)) if use_bias else None

    def __call__(self, x: mx.array) -> mx.array:
        B = int(x.shape[0])
        H, DH = self.num_heads, self.head_dim
        # Reshape to (B, H, DH)
        x3 = x.reshape(B, H, DH)
        # Compute mean/var over DH per head
        mean = mx.mean(x3, axis=-1, keepdims=True)
        var = mx.mean((x3 - mean) * (x3 - mean), axis=-1, keepdims=True)
        xhat = (x3 - mean) * mx.rsqrt(var + self.eps)
        if self.weight is not None:
            xhat = xhat * self.weight
        if self.bias is not None:
            xhat = xhat + self.bias
        return xhat.reshape(B, H * DH)
