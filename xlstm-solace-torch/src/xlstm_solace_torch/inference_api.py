"""High-level inference API for xLSTM-Solace-Torch with Metal acceleration."""

import torch
import warnings
from typing import List, Optional, Union, Dict, Any
from .models import xLSTMSolaceTorch
from .api import create_xlstm_model, create_xlstm_7b_model


def create_inference_model(
    vocab_size: int,
    num_layers: int,
    inp_dim: int,
    head_dim: int,
    head_num: int,
    config_name: str = "default_metal",
) -> xLSTMSolaceTorch:
    """Create a model optimized for inference with Metal acceleration.
    
    Args:
        vocab_size: Size of vocabulary
        num_layers: Number of transformer blocks
        inp_dim: Input/embedding dimension
        head_dim: Dimension per attention head
        head_num: Number of attention heads
        config_name: Configuration name to use
        
    Returns:
        Model ready for inference
    """
    model = create_xlstm_model(
        vocab_size=vocab_size,
        num_layers=num_layers,
        inp_dim=inp_dim,
        head_dim=head_dim,
        head_num=head_num,
        config_name=config_name,
    )
    
    # Ensure inference mode
    model.eval()
    return model


def generate_tokens(
    model: xLSTMSolaceTorch,
    input_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = True,
) -> List[int]:
    """Generate tokens using xLSTM model with Metal acceleration.
    
    Args:
        model: xLSTMSolaceTorch model in inference mode
        input_tokens: Input token tensor of shape (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative)
        top_k: If set, only sample from top k tokens
        do_sample: If True, sample from distribution; if False, use greedy
        
    Returns:
        List of generated token IDs
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure input is on correct device
    if input_tokens.device != device:
        input_tokens = input_tokens.to(device)
    
    # Validate model configuration
    if model.config.mode != "inference":
        warnings.warn("Model should be in inference mode for optimal generation")
    
    generated_tokens = []
    current_input = input_tokens
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass - handle both tuple and single return
            output = model(current_input)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None and top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample or greedy decode
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_tokens.append(int(next_token[0]))
            
            # Append token for next iteration
            current_input = torch.cat([current_input, next_token], dim=1)
    
    return generated_tokens


def quick_inference_test(
    vocab_size: int = 1000,
    prompt_tokens: Optional[List[int]] = None,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """Quick inference test with a small model.
    
    Args:
        vocab_size: Vocabulary size for test model
        prompt_tokens: Input tokens (default: [1, 2, 3, 4])
        max_new_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with test results
    """
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # Ensure good chunk alignment
    
    # Create small test model
    model = create_inference_model(
        vocab_size=vocab_size,
        num_layers=1,
        inp_dim=128,
        head_dim=32,
        head_num=4,
    )
    
    # Convert to tensor
    input_tensor = torch.tensor([prompt_tokens], device='mps')
    
    # Generate tokens
    generated = generate_tokens(
        model=model,
        input_tokens=input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        do_sample=True,
    )
    
    return {
        'input_tokens': prompt_tokens,
        'generated_tokens': generated,
        'total_tokens': len(prompt_tokens) + len(generated),
        'model_config': {
            'vocab_size': vocab_size,
            'num_layers': 1,
            'inp_dim': 128,
            'device': str(next(model.parameters()).device),
            'mode': model.config.mode,
            'kernels': {
                'chunkwise': model.config.chunkwise_kernel,
                'sequence': model.config.sequence_kernel,
                'step': model.config.step_kernel,
            }
        }
    }


# Export main functions
__all__ = [
    'create_inference_model',
    'generate_tokens', 
    'quick_inference_test',
]
