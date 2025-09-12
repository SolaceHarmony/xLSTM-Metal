"""High-level inference API for xLSTM-Solace-MLX models."""

from typing import Optional, Union, List, Dict, Any
import mlx.core as mx
from .model import xLSTMSolaceMLX
from .api import create_xlstm_model
import json


class xLSTMMLXInference:
    """High-level inference wrapper for xLSTM-Solace-MLX models."""
    
    def __init__(
        self, 
        model: Optional[xLSTMSolaceMLX] = None,
        config_name: str = "xlstm_7b_metal"
    ):
        """Initialize MLX inference wrapper.
        
        Args:
            model: Pre-created model (optional)
            config_name: Configuration to use if creating new model
        """
        if model is not None:
            self.model = model
        else:
            if config_name == "xlstm_7b_metal":
                # Create 7B model
                self.model = create_xlstm_model(
                    vocab_size=50304,
                    num_layers=32,
                    inp_dim=4096,
                    head_dim=512,
                    head_num=8
                )
            else:
                # For custom configs, use reasonable defaults
                self.model = create_xlstm_model(
                    vocab_size=50304,
                    num_layers=32,
                    inp_dim=4096,
                    head_dim=512,
                    head_num=8
                )
    
    def load_weights(self, weights_path: str, strict: bool = False):
        """Load model weights using MLX's native loading.
        
        Args:
            weights_path: Path to weights file (.safetensors or .npz)
            strict: Whether to require exact parameter matching
        """
        # MLX can load safetensors directly
        self.model.load_weights(weights_path, strict=strict)
        print(f"✅ Loaded weights from {weights_path} (MLX native)")
    
    def load_weights_from_directory(self, weights_dir: str, strict: bool = False):
        """Load weights from a directory with multiple shards.
        
        Args:
            weights_dir: Directory containing weight files
            strict: Whether to require exact parameter matching
        """
        from pathlib import Path
        
        # Find all .safetensors files in directory
        weight_files = list(Path(weights_dir).glob("*.safetensors"))
        weight_files.sort()  # Load in order
        
        for weight_file in weight_files:
            print(f"Loading shard: {weight_file.name}")
            self.model.load_weights(str(weight_file), strict=False)
        
        print(f"✅ Loaded {len(weight_files)} weight shards from {weights_dir}")
    
    def generate(
        self,
        prompt: Optional[Union[str, mx.array]] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        bos_token_id: int = 0,
        **kwargs
    ) -> mx.array:
        """Generate text tokens using MLX.
        
        Args:
            prompt: Input prompt (string or token array)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            bos_token_id: Beginning of sequence token ID
            
        Returns:
            Generated token array
        """
        # Handle prompt input
        if prompt is None:
            current_tokens = mx.full((1, 1), bos_token_id, dtype=mx.int32)
        elif isinstance(prompt, str):
            # TODO: Add tokenizer support
            raise NotImplementedError("String prompts require tokenizer integration")
        else:
            current_tokens = prompt
            if current_tokens.ndim == 1:
                current_tokens = current_tokens[None, :]  # Add batch dimension
        
        generated_tokens = []
        
        for _ in range(max_length):
            # Forward pass
            logits, _ = self.model(current_tokens)  # Get logits from last position
            next_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Sampling
            if temperature == 0.0:
                # Greedy sampling
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            else:
                # Temperature sampling
                probs = mx.softmax(next_logits, axis=-1)
                
                if top_p < 1.0:
                    # Top-p sampling
                    sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]  # Sort descending
                    sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
                    
                    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
                    top_p_mask = cumsum_probs <= top_p
                    
                    # Ensure at least one token is kept
                    top_p_mask = mx.logical_or(top_p_mask, mx.arange(probs.shape[-1])[None, :] == 0)
                    
                    # Zero out probabilities not in top-p
                    filtered_probs = mx.where(top_p_mask, sorted_probs, 0.0)
                    filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)
                    
                    # Sample from filtered distribution
                    next_token = mx.random.categorical(mx.log(filtered_probs + 1e-10))
                    next_token = mx.take_along_axis(sorted_indices, next_token[:, None], axis=-1)
                else:
                    # Standard categorical sampling
                    next_token = mx.random.categorical(mx.log(probs + 1e-10))
                    next_token = next_token[:, None]
            
            generated_tokens.append(next_token)
            
            # Update current tokens for next iteration
            current_tokens = next_token
        
        # Concatenate all generated tokens
        return mx.concatenate(generated_tokens, axis=1)
    
    def generate_streaming(
        self,
        prompt: Optional[Union[str, mx.array]] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        **kwargs
    ):
        """Generate tokens with streaming output.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Individual generated tokens
        """
        # Handle prompt input
        if prompt is None:
            current_tokens = mx.full((1, 1), kwargs.get('bos_token_id', 0), dtype=mx.int32)
        elif isinstance(prompt, str):
            raise NotImplementedError("String prompts require tokenizer integration")
        else:
            current_tokens = prompt
            if current_tokens.ndim == 1:
                current_tokens = current_tokens[None, :]
        
        for _ in range(max_length):
            # Forward pass
            logits, _ = self.model(current_tokens)
            next_logits = logits[:, -1, :]
            
            # Apply temperature and sample
            if temperature == 0.0:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            else:
                next_logits = next_logits / temperature
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = next_token[:, None]
            
            yield next_token
            
            # Update for next iteration
            current_tokens = next_token
    
    def batch_generate(
        self,
        prompts: List[Union[str, mx.array]],
        max_length: int = 128,
        **kwargs
    ) -> List[mx.array]:
        """Generate tokens for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum tokens to generate per prompt
            
        Returns:
            List of generated token arrays
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt=prompt, max_length=max_length, **kwargs)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        total_params = sum(p.size for p in self.model.parameters())
        return {
            "model_type": "xLSTM-Solace-MLX",
            "backend": "Metal (MLX)",
            "total_parameters": total_params,
            "vocab_size": getattr(self.model, 'vocab_size', 'Unknown'),
            "embedding_dim": getattr(self.model, 'embedding_dim', 'Unknown'),
            "num_blocks": getattr(self.model, 'num_blocks', 'Unknown'),
            "num_heads": getattr(self.model, 'num_heads', 'Unknown'),
        }


def create_mlx_inference_session(
    weights_path: Optional[str] = None,
    weights_dir: Optional[str] = None,
    config_name: str = "xlstm_7b_metal"
) -> xLSTMMLXInference:
    """Create a ready-to-use MLX inference session.
    
    Args:
        weights_path: Path to single weight file (optional)
        weights_dir: Directory with multiple weight shards (optional)
        config_name: Model configuration to use
        
    Returns:
        xLSTMMLXInference instance ready for generation
    """
    inference = xLSTMMLXInference(config_name=config_name)
    
    if weights_path:
        inference.load_weights(weights_path, strict=False)
    elif weights_dir:
        inference.load_weights_from_directory(weights_dir, strict=False)
    
    return inference
