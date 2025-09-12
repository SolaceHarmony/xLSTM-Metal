"""Configuration loading utilities for xLSTM-Solace-Torch."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Structured model configuration."""
    # Device settings
    device: str = "mps"
    force_mps: bool = True
    allow_cpu_fallback: bool = False
    validate_metal_only: bool = True
    
    # Kernel configuration
    chunkwise_kernel: str = "chunkwise--metal_autograd"
    sequence_kernel: str = "native_sequence__metal"
    step_kernel: str = "metal"
    force_metal_kernels: bool = True
    
    # Model architecture
    embedding_dim: int = 4096
    num_heads: int = 8
    num_blocks: int = 32
    vocab_size: int = 50304
    head_dim: Optional[int] = None
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    add_out_norm: bool = True
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    chunk_size: int = 64
    mode: str = "train"
    return_last_states: bool = True


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return Path(__file__).parent / "configs"


def load_config(config_name: str = "default_metal") -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_name: Name of config file (without .json extension)
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = get_config_dir() / f"{config_name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    
    # Validate required Metal settings
    _validate_metal_config(config, config_name)
    
    return config


def _validate_metal_config(config: Dict[str, Any], config_name: str) -> None:
    """Validate that configuration enforces Metal-only acceleration."""
    issues = []
    
    # Check device settings
    device_settings = config.get("device_settings", {})
    if device_settings.get("device") != "mps":
        issues.append("device must be 'mps'")
    if not device_settings.get("force_mps", False):
        issues.append("force_mps must be true")
    if device_settings.get("allow_cpu_fallback", True):
        issues.append("allow_cpu_fallback must be false")
    
    # Check kernel configuration
    kernel_config = config.get("kernel_configuration", {})
    if "native" in kernel_config.get("chunkwise_kernel", "") and "metal" not in kernel_config.get("chunkwise_kernel", ""):
        issues.append("chunkwise_kernel must use Metal acceleration")
    if kernel_config.get("step_kernel") == "native":
        issues.append("step_kernel cannot be 'native' (use 'metal')")
    if "native" in kernel_config.get("sequence_kernel", "") and "metal" not in kernel_config.get("sequence_kernel", ""):
        issues.append("sequence_kernel must use Metal acceleration")
    
    if issues:
        raise ValueError(f"Configuration '{config_name}' violates Metal-only requirements:\n" + 
                        "\n".join(f"  - {issue}" for issue in issues))


def create_model_config_from_dict(config_dict: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from configuration dictionary."""
    # Merge device settings
    device_settings = config_dict.get("device_settings", {})
    kernel_config = config_dict.get("kernel_configuration", {})
    model_defaults = config_dict.get("model_defaults", config_dict.get("model_7b", {}))
    
    return ModelConfig(
        # Device settings
        device=device_settings.get("device", "mps"),
        force_mps=device_settings.get("force_mps", True),
        allow_cpu_fallback=device_settings.get("allow_cpu_fallback", False),
        validate_metal_only=device_settings.get("validate_metal_only", True),
        
        # Kernel configuration
        chunkwise_kernel=kernel_config.get("chunkwise_kernel", "chunkwise--metal_autograd"),
        sequence_kernel=kernel_config.get("sequence_kernel", "native_sequence__metal"),
        step_kernel=kernel_config.get("step_kernel", "metal"),
        force_metal_kernels=kernel_config.get("force_metal_kernels", True),
        
        # Model architecture
        embedding_dim=model_defaults.get("embedding_dim", 4096),
        num_heads=model_defaults.get("num_heads", 8),
        num_blocks=model_defaults.get("num_blocks", 32),
        vocab_size=model_defaults.get("vocab_size", 50304),
        head_dim=model_defaults.get("head_dim"),
        use_bias=model_defaults.get("use_bias", False),
        norm_eps=model_defaults.get("norm_eps", 1e-6),
        norm_reduction_force_float32=model_defaults.get("norm_reduction_force_float32", True),
        add_out_norm=model_defaults.get("add_out_norm", True),
        qk_dim_factor=model_defaults.get("qk_dim_factor", 0.5),
        v_dim_factor=model_defaults.get("v_dim_factor", 1.0),
        gate_soft_cap=model_defaults.get("gate_soft_cap", 15.0),
        output_logit_soft_cap=model_defaults.get("output_logit_soft_cap", 30.0),
        chunk_size=model_defaults.get("chunk_size", 64),
        mode=model_defaults.get("mode", "inference"),
        return_last_states=model_defaults.get("return_last_states", True),
    )


def load_model_config(config_name: str = "default_metal") -> ModelConfig:
    """Load and create ModelConfig from configuration file."""
    config_dict = load_config(config_name)
    return create_model_config_from_dict(config_dict)


def list_available_configs() -> list[str]:
    """List all available configuration files."""
    config_dir = get_config_dir()
    return [f.stem for f in config_dir.glob("*.json")]
