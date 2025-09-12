"""Kernel validation utilities to ensure Metal-only acceleration."""

def validate_metal_only_kernels(config):
    """Validate that configuration uses only Metal-accelerated kernels.
    
    Args:
        config: xLSTMSolaceTorchConfig instance
        
    Raises:
        ValueError: If any kernel is not Metal-accelerated
    """
    issues = []
    
    # Check chunkwise kernel
    if hasattr(config, 'chunkwise_kernel'):
        if 'native' in config.chunkwise_kernel and 'metal' not in config.chunkwise_kernel:
            issues.append(f"chunkwise_kernel '{config.chunkwise_kernel}' is not Metal-accelerated")
    
    # Check sequence kernel  
    if hasattr(config, 'sequence_kernel'):
        if 'native' in config.sequence_kernel and 'metal' not in config.sequence_kernel:
            issues.append(f"sequence_kernel '{config.sequence_kernel}' is not Metal-accelerated")
    
    # Check step kernel
    if hasattr(config, 'step_kernel'):
        if config.step_kernel == 'native':
            issues.append(f"step_kernel '{config.step_kernel}' is not Metal-accelerated")
    
    if issues:
        raise ValueError(
            "Configuration uses non-Metal kernels (forbidden):\n" + 
            "\n".join(f"  - {issue}" for issue in issues) +
            "\n\nThis codebase requires Metal acceleration. Use metal_autograd, metal_custbw, or native_sequence__metal kernels."
        )

def get_recommended_metal_kernels():
    """Get recommended Metal-accelerated kernel configuration."""
    return {
        'chunkwise_kernel': 'chunkwise--metal_autograd',
        'sequence_kernel': 'native_sequence__metal',
        'step_kernel': 'metal',
    }
