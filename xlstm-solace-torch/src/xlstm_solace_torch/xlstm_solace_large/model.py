"""
xLSTM Solace Large Model for Apple Silicon.

Based on official xlstm.xlstm_large.model but uses our existing Apple Metal-optimized
kernels and components. This provides the same API as the official version but with
full Apple Silicon acceleration.
"""

import torch
from torch import nn
from typing import Optional

from .config import xLSTMSolaceLargeConfig
from ..models.model import (
    xLSTMSolaceBlockStack, 
    mLSTMStateType,
    xLSTMSolaceTorchConfig
)
from ..models.components import soft_cap
from ..models.generate import generate_tokens, get_sampling_fn


class xLSTMSolaceLarge(nn.Module):
    """
    Apple Silicon optimized xLSTM Large model.
    
    Provides the same API as official xlstm.xlstm_large.model.xLSTMLarge
    but uses our Metal-accelerated kernels instead of Triton.
    """
    config_class = xLSTMSolaceLargeConfig

    def __init__(self, config: xLSTMSolaceLargeConfig):
        super().__init__()
        self.config = config
        
        # Convert our Apple config to internal format
        internal_config = self._convert_to_internal_config(config)
        
        # Use our existing components with Metal acceleration
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.backbone = xLSTMSolaceBlockStack(internal_config)
        self.lm_head = nn.Linear(
            in_features=config.embedding_dim, 
            out_features=config.vocab_size, 
            bias=False
        )

    def _convert_to_internal_config(self, config: xLSTMSolaceLargeConfig) -> xLSTMSolaceTorchConfig:
        """Convert xLSTMSolaceLargeConfig to our internal xLSTMSolaceTorchConfig."""
        return xLSTMSolaceTorchConfig(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            vocab_size=config.vocab_size,
            use_bias=config.use_bias,
            norm_eps=config.norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            add_out_norm=config.add_out_norm,
            qk_dim_factor=config.qk_dim_factor,
            v_dim_factor=config.v_dim_factor,
            chunkwise_kernel=config.chunkwise_kernel,
            sequence_kernel=config.sequence_kernel,
            step_kernel=config.step_kernel,
            mode=config.mode,
            chunk_size=config.chunk_size,
            return_last_states=config.return_last_states,
            autocast_kernel_dtype=config.autocast_kernel_dtype,
            eps=config.eps,
            inference_state_dtype=config.inference_state_dtype,
            ffn_proj_factor=config.ffn_proj_factor,
            ffn_round_up_to_multiple_of=config.ffn_round_up_to_multiple_of,
            gate_soft_cap=config.gate_soft_cap,
            output_logit_soft_cap=config.output_logit_soft_cap,
            weight_mode=config.weight_mode,
        )

    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[mLSTMStateType] = None
    ) -> torch.Tensor | tuple[torch.Tensor, mLSTMStateType]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, S].
            state: State dictionary of the model. 
                   If None, the state is initialized, the model starts from an empty initial state.
        
        Returns:
            logits: Logits tensor of shape [B, S, V].
            Tuple of logits and state: State dictionary of the model, if `return_last_states` is True.
        """
        assert x.ndim == 2, f"Input must have shape [B, S], got {x.shape}"
        B, S = x.shape

        x = self.embedding(x)
        x, state = self.backbone(x, state)
        logits = self.lm_head(x)
        logits_capped = soft_cap(logits, self.config.output_logit_soft_cap)
        
        if self.config.return_last_states:
            return logits_capped, state
        else:
            return logits_capped

    def generate(
        self,
        prefill_tokens: torch.Tensor,
        max_length: int,
        sampling_type: str = "greedy",
        state: Optional[mLSTMStateType] = None,
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        """
        Generate tokens from the model using Apple Metal acceleration.

        Args:
            prefill_tokens: Tensor of shape [B, S] with the prefill tokens.
            max_length: Maximum length of the generated sequence.
            sampling_type: Sampling type to use, e.g. 'greedy'.
            state: State dictionary of the model. 
                   If None, the state is initialized, the model starts from an empty initial state.
        
        Returns:
            tokens: Generated tokens tensor of shape [B, S].
            state: State dictionary of the model after the last generation step.
        """
        # Use official API pattern exactly
        sampling_fn = get_sampling_fn(sampling_type)
        tokens, state = generate_tokens(
            llm_forward=self.forward,
            prefill_tokens=prefill_tokens,
            max_length=max_length,
            token_sample_fn=sampling_fn,
            state=state,
            device=str(self.embedding.weight.device),
        )
        return tokens, state

    def to_mps(self):
        """Convenience method to move model to MPS device for Apple Silicon."""
        if torch.backends.mps.is_available():
            return self.to('mps')
        else:
            raise RuntimeError("MPS not available on this device")
            
    def enable_metal_acceleration(self):
        """Verify and enable Metal acceleration throughout the model."""
        # This is already enabled by default in our Apple implementation
        # This method is for API compatibility
        if not torch.backends.mps.is_available():
            raise RuntimeError("Metal acceleration requires MPS backend")
        return self
