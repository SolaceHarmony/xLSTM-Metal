#!/usr/bin/env python
"""Generic Weight Loader – MLX Implementation (Universal nn.Module Parameter Loading)

Overview
--------
Provides flexible utilities for loading pretrained weights from safetensors
into any MLX nn.Module, with automatic dtype conversion, key mapping, and
nested parameter traversal.

This module generalizes the weight loading logic used in WiredxLSTM to work
with arbitrary module hierarchies, enabling reuse across different model
architectures without custom loading code.

Design Philosophy
-----------------
**Problem**: Each model class typically implements custom weight loading:
  - Hardcoded parameter paths (e.g., "backbone.blocks.0.mlstm.q_proj.weight")
  - Manual dtype conversion scattered across loaders
  - Duplicate logic for nested modules

**Solution**: Generic loader that:
  - Traverses module hierarchy automatically
  - Maps safetensors keys to module parameters via flexible strategies
  - Handles dtype conversion centrally
  - Supports partial loading and strict validation

Key Features
------------
1. **Automatic Parameter Discovery**
   - Recursively finds all parameters in nn.Module tree
   - Builds flat parameter path → object mapping
   
2. **Flexible Key Mapping**
   - Direct key matching (safetensors_key == param_path)
   - Prefix stripping (remove "model.", "backbone.", etc.)
   - Custom mapping functions (user-defined transformations)
   
3. **Dtype Management**
   - Convert weights to target dtype during load
   - Preserve original dtypes when needed
   - Support mixed-precision loading
   
4. **Validation Modes**
   - Strict: Error if any weight is missing
   - Permissive: Load what exists, skip missing
   - Report: Detailed missing/unexpected keys summary

Usage Patterns
--------------

**Basic Loading (Direct Key Match)**:
  >>> from xlstm_metal.mlx_jit.utils.generic_weight_loader import load_weights_from_safetensors
  >>> model = MyModule()
  >>> load_weights_from_safetensors(
  ...     model, 
  ...     "xlstm_7b_model",
  ...     target_dtype=mx.bfloat16
  ... )

**With Prefix Stripping**:
  >>> # Safetensors has "model.layer1.weight", module expects "layer1.weight"
  >>> load_weights_from_safetensors(
  ...     model,
  ...     "model_dir",
  ...     strip_prefix="model."
  ... )

**Custom Key Mapping**:
  >>> def my_key_mapper(safetensors_key):
  ...     # Transform HF checkpoint keys to local module structure
  ...     return safetensors_key.replace("transformer.", "").replace("h.", "layers.")
  >>> 
  >>> load_weights_from_safetensors(
  ...     model,
  ...     "checkpoint_dir",
  ...     key_mapper=my_key_mapper
  ... )

**Dictionary Loading**:
  >>> from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards
  >>> weights_dict = load_safetensor_shards("model_dir")
  >>> load_weights_from_dict(model, weights_dict, target_dtype=mx.float32)

**Partial Loading with Reporting**:
  >>> missing, unexpected = load_weights_from_safetensors(
  ...     model,
  ...     "checkpoint_dir",
  ...     strict=False,
  ...     return_diagnostics=True
  ... )
  >>> print(f"Missing: {missing}")
  >>> print(f"Unexpected: {unexpected}")

Architecture
------------
Core functions:

1. `get_parameter_dict(module)` → Dict[str, mx.array]
   - Flattens module hierarchy into "path.to.param" → array mapping
   - Example: {"layer1.weight": array, "layer1.bias": array, ...}

2. `load_weights_from_dict(module, weights, ...)` 
   - Loads from in-memory dict (post-safetensors parsing)
   - Handles dtype conversion and key mapping

3. `load_weights_from_safetensors(module, model_dir, ...)`
   - High-level API: reads shards + loads weights
   - Combines load_safetensor_shards + load_weights_from_dict

Parameter Traversal
-------------------
MLX modules store parameters as direct attributes or in nested modules.
We recursively traverse:

  module
    ├── param1 (mx.array)
    ├── param2 (mx.array)
    ├── submodule
    │   ├── param3 (mx.array)
    │   └── subsubmodule
    │       └── param4 (mx.array)
    └── param5 (mx.array)

Flattened paths:
  - "param1", "param2"
  - "submodule.param3"
  - "submodule.subsubmodule.param4"
  - "param5"

Key Mapping Strategies
----------------------
**Direct Match** (default):
  safetensors_key == parameter_path

**Prefix Stripping**:
  safetensors: "backbone.blocks.0.weight"
  strip_prefix: "backbone."
  result: "blocks.0.weight"

**Custom Mapper**:
  def mapper(key):
      # Arbitrary transformation
      return key.replace("old_name", "new_name")

Dtype Conversion
----------------
Weights can be loaded in different dtypes than stored:
  - Safetensors typically store float32 or bfloat16
  - Runtime may want bfloat16 (compute) or float32 (accumulation)
  
The loader applies: mx.array(weight, dtype=target_dtype)

Strict vs Permissive Loading
-----------------------------
**Strict Mode** (strict=True):
  - Every model parameter must have corresponding weight
  - Every safetensors weight must be used
  - Raises ValueError on mismatch

**Permissive Mode** (strict=False):
  - Loads available weights, skips missing
  - Ignores unused safetensors keys
  - Useful for partial checkpoint loading or fine-tuning

Diagnostics
-----------
When return_diagnostics=True, returns (missing_keys, unexpected_keys):
  - missing_keys: Model parameters not found in weights
  - unexpected_keys: Safetensors keys not used by model

Useful for debugging checkpoint compatibility issues.

Weight Tying
------------
Some models share parameters (e.g., embedding.weight = lm_head.weight).
This loader does NOT handle weight tying automatically - models should
implement _apply_weight_tying() post-load if needed.

MLX Array Semantics
-------------------
MLX arrays are immutable views. When loading:
  module.param = mx.array(weight, dtype=target_dtype)

This creates a new array object with converted dtype. Original safetensors
data is unaffected.

Comparison to PyTorch
---------------------
PyTorch: model.load_state_dict(torch.load("checkpoint.pt"))
MLX: load_weights_from_safetensors(model, "checkpoint_dir")

Key differences:
  - MLX uses safetensors (safer, faster) vs pickle (security risk)
  - MLX requires explicit dtype conversion vs PyTorch auto-matching
  - MLX shards are separate files vs single .pt file

Examples
--------
**Complete Model Loading**:
  >>> import mlx.nn as nn
  >>> import mlx.core as mx
  >>> from xlstm_metal.mlx_jit.utils.generic_weight_loader import load_weights_from_safetensors
  >>> 
  >>> class MyModel(nn.Module):
  ...     def __init__(self):
  ...         super().__init__()
  ...         self.layer1 = nn.Linear(512, 256)
  ...         self.layer2 = nn.Linear(256, 128)
  ... 
  >>> model = MyModel()
  >>> load_weights_from_safetensors(
  ...     model,
  ...     "pretrained_weights",
  ...     target_dtype=mx.bfloat16,
  ...     strict=True
  ... )

**Partial Loading for Fine-Tuning**:
  >>> # Load pretrained encoder, skip decoder head
  >>> load_weights_from_safetensors(
  ...     model,
  ...     "pretrained_encoder",
  ...     strict=False
  ... )
  >>> # Decoder randomly initialized, ready for task-specific training

**Key Mapping for HuggingFace Models**:
  >>> def hf_to_local(key):
  ...     # HF uses "transformer.h.{i}" we use "layers.{i}"
  ...     return key.replace("transformer.h.", "layers.").replace("transformer.", "")
  >>> 
  >>> load_weights_from_safetensors(
  ...     model,
  ...     "hf_checkpoint",
  ...     key_mapper=hf_to_local
  ... )

Integration with Existing Code
-------------------------------
WiredxLSTM currently uses custom _load_weights_from_dict method.
Could be refactored to:

  def _load_weights_from_dict(self, weights_dict):
      load_weights_from_dict(
          self,
          weights_dict,
          target_dtype=self.compute_dtype,
          strict=True
      )
      self._apply_weight_tying()

Parity
------
Generic design allows usage across MLX models, not specific to xLSTM.
Logic can be adapted for PyTorch equivalents with minimal changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


def get_parameter_dict(module: nn.Module, prefix: str = "") -> Dict[str, mx.array]:
    """Recursively extract all parameters from an nn.Module as a flat dict.
    
    Traverses the module tree and builds a mapping from parameter path
    (dotted notation) to the actual mx.array parameter object.
    
    Parameters
    ----------
    module : nn.Module
        Root module to extract parameters from.
    prefix : str, default ""
        Internal parameter for recursion (dotted path prefix).
    
    Returns
    -------
    param_dict : dict[str, mx.array]
        Flat mapping of "path.to.param" → mx.array.
        
    Examples
    --------
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 5),
    ...     nn.Linear(5, 2)
    ... )
    >>> params = get_parameter_dict(model)
    >>> list(params.keys())
    ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias']
    
    Notes
    -----
    - Only includes mx.array attributes (parameters), not submodules
    - Uses dot notation for nested paths: "encoder.layer1.weight"
    - Recursively processes all children modules
    """
    params = {}

    # Iterate through all attributes
    for name, value in vars(module).items():
        # Skip private attributes
        if name.startswith('_'):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        # If it's an mx.array, it's a parameter
        if isinstance(value, mx.array):
            params[full_name] = value
        # If it's a module, recurse
        elif isinstance(value, nn.Module):
            params.update(get_parameter_dict(value, prefix=full_name))
        # If it's a list of modules, handle each
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, nn.Module):
                    params.update(get_parameter_dict(item, prefix=f"{full_name}.{i}"))

    return params


def set_parameter(module: nn.Module, param_path: str, value: mx.array) -> bool:
    """Set a parameter in a nested module using dotted path notation.
    
    Traverses module hierarchy following the path and sets the final
    attribute to the provided value.
    
    Parameters
    ----------
    module : nn.Module
        Root module containing the parameter.
    param_path : str
        Dotted path to parameter (e.g., "layer1.weight").
    value : mx.array
        New parameter value.
    
    Returns
    -------
    success : bool
        True if parameter was set successfully, False if path not found.
        
    Examples
    --------
    >>> model = nn.Linear(10, 5)
    >>> new_weight = mx.zeros((5, 10))
    >>> set_parameter(model, "weight", new_weight)
    True
    >>> model.weight.shape
    [5, 10]
    
    >>> set_parameter(model, "nonexistent.weight", new_weight)
    False
    """
    parts = param_path.split('.')
    obj = module

    # Navigate to parent object
    for part in parts[:-1]:
        try:
            # Handle list/tuple indices
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        except (AttributeError, IndexError, KeyError):
            return False

    # Set final attribute
    final_attr = parts[-1]
    try:
        if hasattr(obj, final_attr):
            setattr(obj, final_attr, value)
            return True
    except Exception:
        pass

    return False


def load_weights_from_dict(
        module: nn.Module,
        weights_dict: Dict[str, mx.array],
        target_dtype: Optional[mx.Dtype] = None,
        strip_prefix: Optional[str] = None,
        key_mapper: Optional[Callable[[str], str]] = None,
        strict: bool = True,
        return_diagnostics: bool = False,
) -> Union[None, Tuple[List[str], List[str]]]:
    """Load weights from a dictionary into an nn.Module.
    
    Generic weight loader that handles key mapping, dtype conversion,
    and validation for any MLX module structure.
    
    Parameters
    ----------
    module : nn.Module
        Target module to load weights into.
    weights_dict : dict[str, mx.array]
        Dictionary of weights (typically from load_safetensor_shards).
    target_dtype : mx.Dtype | None, optional
        Convert all weights to this dtype. If None, preserves original dtype.
    strip_prefix : str | None, optional
        Remove this prefix from safetensors keys before matching.
        Example: strip_prefix="backbone." converts "backbone.layer1.weight" 
        to "layer1.weight".
    key_mapper : callable | None, optional
        Custom function to transform safetensors keys to module parameter paths.
        Called as: module_key = key_mapper(safetensors_key).
        Applied after strip_prefix if both provided.
    strict : bool, default True
        If True, raises error if any model parameters are missing or any
        safetensors keys are unused. If False, loads what matches and
        ignores mismatches.
    return_diagnostics : bool, default False
        If True, returns (missing_keys, unexpected_keys) instead of None.
        
    Returns
    -------
    None | (missing_keys, unexpected_keys)
        If return_diagnostics=False: None
        If return_diagnostics=True: tuple of lists containing missing and 
        unexpected keys for debugging.
        
    Raises
    ------
    ValueError
        In strict mode, if any required parameters are missing or unexpected
        keys found.
        
    Examples
    --------
    >>> # Basic loading with dtype conversion
    >>> weights = load_safetensor_shards("model_dir")
    >>> load_weights_from_dict(model, weights, target_dtype=mx.bfloat16)
    
    >>> # Strip HuggingFace "model." prefix
    >>> load_weights_from_dict(model, weights, strip_prefix="model.")
    
    >>> # Custom key mapping
    >>> def my_mapper(key):
    ...     return key.replace("old_", "new_")
    >>> load_weights_from_dict(model, weights, key_mapper=my_mapper)
    
    >>> # Permissive loading with diagnostics
    >>> missing, unexpected = load_weights_from_dict(
    ...     model, weights, strict=False, return_diagnostics=True
    ... )
    """
    # Get all module parameters
    module_params = get_parameter_dict(module)

    # Build reverse mapping: safetensors_key → module_param_path
    weights_to_params = {}
    used_weights = set()

    for safetensors_key in weights_dict.keys():
        # Apply transformations to get module parameter path
        module_key = safetensors_key

        # Strip prefix if provided
        if strip_prefix and module_key.startswith(strip_prefix):
            module_key = module_key[len(strip_prefix):]

        # Apply custom mapper if provided
        if key_mapper is not None:
            module_key = key_mapper(module_key)

        weights_to_params[safetensors_key] = module_key

    # Load matching weights
    loaded_params = set()

    for safetensors_key, module_key in weights_to_params.items():
        if module_key in module_params:
            # Get weight tensor
            weight = weights_dict[safetensors_key]

            # Apply dtype conversion if requested
            if target_dtype is not None:
                weight = mx.array(weight, dtype=target_dtype)

            # Set parameter in module
            if set_parameter(module, module_key, weight):
                loaded_params.add(module_key)
                used_weights.add(safetensors_key)

    # Validation
    all_module_params = set(module_params.keys())
    missing_keys = all_module_params - loaded_params
    unexpected_keys = set(weights_dict.keys()) - used_weights

    if strict:
        if missing_keys:
            raise ValueError(
                f"Missing {len(missing_keys)} required parameters in checkpoint:\n"
                f"{sorted(list(missing_keys)[:10])}{'...' if len(missing_keys) > 10 else ''}"
            )
        if unexpected_keys:
            raise ValueError(
                f"Found {len(unexpected_keys)} unexpected keys in checkpoint:\n"
                f"{sorted(list(unexpected_keys)[:10])}{'...' if len(unexpected_keys) > 10 else ''}"
            )

    if return_diagnostics:
        return sorted(list(missing_keys)), sorted(list(unexpected_keys))

    return None


def load_weights_from_safetensors(
        module: nn.Module,
        model_dir: Union[str, Path],
        index_filename: str = "model.safetensors.index.json",
        target_dtype: Optional[mx.Dtype] = None,
        strip_prefix: Optional[str] = None,
        key_mapper: Optional[Callable[[str], str]] = None,
        strict: bool = True,
        return_diagnostics: bool = False,
) -> Union[None, Tuple[List[str], List[str]]]:
    """Load weights from safetensors checkpoint into an nn.Module.
    
    High-level API that combines shard loading and weight loading in one call.
    Reads all safetensors shards from a HuggingFace-style checkpoint directory
    and loads them into the module with optional transformations.
    
    Parameters
    ----------
    module : nn.Module
        Target module to load weights into.
    model_dir : str | Path
        Directory containing safetensors files and index.
    index_filename : str, default "model.safetensors.index.json"
        Name of the HuggingFace shard index file.
    target_dtype : mx.Dtype | None, optional
        Convert all weights to this dtype.
    strip_prefix : str | None, optional
        Remove this prefix from checkpoint keys before matching.
    key_mapper : callable | None, optional
        Custom key transformation function.
    strict : bool, default True
        Enforce exact parameter matching.
    return_diagnostics : bool, default False
        Return missing/unexpected keys instead of None.
        
    Returns
    -------
    None | (missing_keys, unexpected_keys)
        None on success, or diagnostic lists if requested.
        
    Raises
    ------
    FileNotFoundError
        If model_dir, index file, or shard files not found.
    ValueError
        In strict mode, if parameters missing or unexpected.
        
    Examples
    --------
    >>> # Simple loading
    >>> model = MyModel()
    >>> load_weights_from_safetensors(model, "xlstm_7b_model")
    
    >>> # Mixed precision
    >>> load_weights_from_safetensors(
    ...     model, 
    ...     "checkpoint_dir",
    ...     target_dtype=mx.bfloat16
    ... )
    
    >>> # Debug checkpoint compatibility
    >>> missing, unexpected = load_weights_from_safetensors(
    ...     model,
    ...     "checkpoint_dir",
    ...     strict=False,
    ...     return_diagnostics=True
    ... )
    >>> if missing:
    ...     print(f"Warning: {len(missing)} parameters not loaded")
    """
    # Import here to avoid circular dependency
    from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards

    model_dir = Path(model_dir)

    # Load all shards into memory
    weights_dict = load_safetensor_shards(str(model_dir), index_filename=index_filename)

    # Load into module
    return load_weights_from_dict(
        module=module,
        weights_dict=weights_dict,
        target_dtype=target_dtype,
        strip_prefix=strip_prefix,
        key_mapper=key_mapper,
        strict=strict,
        return_diagnostics=return_diagnostics,
    )


__all__ = [
    'get_parameter_dict',
    'set_parameter',
    'load_weights_from_dict',
    'load_weights_from_safetensors',
]
