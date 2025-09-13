"""High-level inference API for xLSTM-Solace-Torch models."""

from typing import Optional, Union, List, Dict, Any
import torch
from .models import xLSTMTorch
from .models.generate import generate_tokens, get_sampling_fn
from .api import create_xlstm_7b_model, create_xlstm_model
import json


class xLSTMInference:
    """High-level inference wrapper for xLSTM-Solace-Torch models."""
    
    def __init__(
        self, 
        model: Optional[xLSTMTorch] = None,
        config_name: str = "xlstm_7b_metal",
        device: str = "mps"
    ):
        """Initialize inference wrapper.
        
        Args:
            model: Pre-created model (optional)
            config_name: Configuration to use if creating new model
            device: Device to run inference on (must be 'mps')
        """
        if model is not None:
            self.model = model
        else:
            if config_name == "xlstm_7b_metal":
                self.model = create_xlstm_7b_model(config_name=config_name, device=device)
            else:
                # For custom configs, we need basic parameters
                self.model = create_xlstm_model(
                    vocab_size=50304,  # Default 7B vocab size
                    num_layers=32,
                    inp_dim=4096,
                    head_dim=512,
                    head_num=8,
                    config_name=config_name,
                    device=device
                )
        
        self.device = device
        self.model.eval()  # Set to evaluation mode
    
    def load_weights(self, weights_path: str, strict: bool = False):
        """Load model weights from safetensors or state dict.
        
        Args:
            weights_path: Path to weights file (.safetensors or .pth)
            strict: Whether to require exact parameter matching
        """
        if weights_path.endswith('.safetensors'):
            # Use safetensors loading
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device=str(self.device))
            self.model.load_state_dict(state_dict, strict=strict)
        else:
            # Use PyTorch loading
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=strict)
        
        print(f"✅ Loaded weights from {weights_path}")
    
    def generate(
        self,
        prompt: Optional[Union[str, torch.Tensor]] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        sampling_strategy: str = "greedy",
        bos_token_id: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """Generate text tokens.
        
        Args:
            prompt: Input prompt (string or token tensor)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (not implemented yet)
            sampling_strategy: Sampling strategy ('greedy')
            bos_token_id: Beginning of sequence token ID
            
        Returns:
            Generated token tensor
        """
        # Handle prompt input
        if prompt is None:
            prefill_tokens = None
        elif isinstance(prompt, str):
            # TODO: Add tokenizer support
            raise NotImplementedError("String prompts require tokenizer integration")
        else:
            prefill_tokens = prompt.to(self.device)
        
        # Get sampling function
        sample_fn = get_sampling_fn(sampling_strategy)
        
        # Create forward function wrapper
        def llm_forward(tokens: torch.Tensor, state=None):
            # Ensure tokens are on correct device
            tokens = tokens.to(self.device)
            
            # Forward pass through model
            with torch.no_grad():
                output = self.model(tokens)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    logits = output[0]  # Assume first element is logits
                    new_state = output[1] if len(output) > 1 else state
                else:
                    logits = output
                    new_state = state
                
                return logits, new_state
        
        # Generate tokens
        generated_tokens, final_state = generate_tokens(
            llm_forward=llm_forward,
            prefill_tokens=prefill_tokens,
            max_length=max_length,
            token_sample_fn=sample_fn,
            bos_token_id=bos_token_id,
            device=str(self.device),
            **kwargs
        )
        
        return generated_tokens
    
    def generate_streaming(
        self,
        prompt: Optional[Union[str, torch.Tensor]] = None,
        max_length: int = 128,
        **kwargs
    ):
        """Generate tokens with streaming output.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            
        Yields:
            Individual generated tokens
        """
        # For streaming, we generate one token at a time
        if prompt is None:
            current_tokens = torch.full((1, 1), fill_value=kwargs.get('bos_token_id', 0), 
                                      dtype=torch.long, device=self.device)
        elif isinstance(prompt, str):
            raise NotImplementedError("String prompts require tokenizer integration")
        else:
            current_tokens = prompt.to(self.device)
        
        state = None
        
        for _ in range(max_length):
            # Generate one step
            def llm_forward(tokens: torch.Tensor, state=None):
                tokens = tokens.to(self.device)
                with torch.no_grad():
                    output = self.model(tokens)
                    if isinstance(output, tuple):
                        logits = output[0]
                        new_state = output[1] if len(output) > 1 else state
                    else:
                        logits = output
                        new_state = state
                    return logits, new_state
            
            logits, state = llm_forward(current_tokens, state)
            
            # Sample next token
            sample_fn = get_sampling_fn(kwargs.get('sampling_strategy', 'greedy'))
            next_token = sample_fn(logits[:, -1:])
            
            yield next_token
            
            # Update current tokens for next iteration
            current_tokens = next_token
    
    def batch_generate(
        self,
        prompts: List[Union[str, torch.Tensor]],
        max_length: int = 128,
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate tokens for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum tokens to generate per prompt
            
        Returns:
            List of generated token tensors
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt=prompt, max_length=max_length, **kwargs)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_type": "xLSTM-Solace-Torch",
            "device": str(self.device),
            "total_parameters": total_params,
            "vocab_size": self.model.config.vocab_size,
            "embedding_dim": self.model.config.embedding_dim,
            "num_blocks": self.model.config.num_blocks,
            "num_heads": self.model.config.num_heads,
            "kernels": {
                "chunkwise": self.model.config.chunkwise_kernel,
                "sequence": self.model.config.sequence_kernel,
                "step": self.model.config.step_kernel,
            }
        }


def create_inference_session(
    weights_path: Optional[str] = None,
    config_name: str = "xlstm_7b_metal",
    device: str = "mps"
) -> xLSTMInference:
    """Create a ready-to-use inference session.
    
    Args:
        weights_path: Path to model weights (optional)
        config_name: Model configuration to use
        device: Device for inference
        
    Returns:
        xLSTMInference instance ready for generation
    """
    inference = xLSTMInference(config_name=config_name, device=device)
    
    if weights_path:
        inference.load_weights(weights_path, strict=False)
    
    return inference
