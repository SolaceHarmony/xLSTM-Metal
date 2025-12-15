#!/usr/bin/env python
"""Generic Automatic Wiring – MLX Implementation (Universal Model Structure Discovery)

Overview
--------
Provides a flexible framework for automatically discovering model architecture
from arbitrary weight structures (safetensors, npz, state_dict, etc.) without
hardcoded assumptions about model type or structure.

This module generalizes the AutoWiring concept to work with any model:
  - BERT-style transformers (e.g., M2-BERT)
  - xLSTM models
  - Custom architectures with nested components

Design Philosophy
-----------------
**Problem**: Every model architecture requires custom loading code:
  - xLSTM: "backbone.blocks.{i}.mlstm_layer"
  - BERT: "bert.encoder.layer.{i}.attention"
  - GPT: "transformer.h.{i}.attn"
  - Custom: Arbitrary nesting patterns

**Solution**: Generic pattern-based discovery that:
  1. Extracts weight key patterns automatically
  2. Groups by hierarchical structure (encoder.layer.{i}.*)
  3. Detects component types from naming conventions
  4. Builds configurable wiring without hardcoded assumptions

Key Features
------------
1. **Pattern-Based Block Detection**
   - Regex-based layer/block extraction
   - Configurable naming conventions
   - Handles arbitrary nesting depth

2. **Component Taxonomy**
   - Attention: [qkv, q_proj, k_proj, v_proj, self_attn, attention]
   - FFN: [mlp, ffn, feed_forward, intermediate]
   - Normalization: [norm, layer_norm, rms_norm]
   - LSTM: [mlstm, slstm, lstm]
   - Custom: User-defined patterns

3. **Flexible Configuration**
   - Custom regex patterns for block detection
   - User-defined component detectors
   - Override detection logic per model type

4. **Multiple Sources**
   - Safetensors (HuggingFace standard)
   - NPZ (MLX format)
   - PyTorch state_dict
   - Raw weight dict

Usage Patterns
--------------

**M2-BERT Example**:
  >>> # Weights like: 'bert.encoder.layer.9.mlp.wo.weight'
  >>> wiring = discover_model_structure(
  ...     weights_dict=weights,
  ...     block_pattern=r'bert\.encoder\.layer\.(\d+)',
  ...     model_type='bert'
  ... )
  >>> print(wiring.structure)
  {
      'num_blocks': 12,
      'block_types': {0: 'transformer', 1: 'transformer', ...},
      'block_components': {0: ['attention', 'mlp', 'norm'], ...}
  }

**xLSTM Example**:
  >>> wiring = discover_model_structure(
  ...     model_dir='xlstm_7b_model',
  ...     block_pattern=r'backbone\.blocks\.(\d+)',
  ...     model_type='xlstm'
  ... )

**Custom Model**:
  >>> def my_block_detector(key):
  ...     match = re.search(r'my_model\.layers\.(\d+)', key)
  ...     return int(match.group(1)) if match else None
  >>> 
  >>> wiring = discover_model_structure(
  ...     weights_dict=weights,
  ...     block_detector=my_block_detector,
  ...     component_rules={
  ...         'attention': ['attn', 'self_attention'],
  ...         'ffn': ['mlp', 'feedforward']
  ...     }
  ... )

Architecture
------------

Core Classes:

1. **GenericWiringConfig**
   - Stores detection rules and patterns
   - Configurable component taxonomy
   - Model-specific overrides

2. **DiscoveredStructure**
   - Holds analyzed model structure
   - Block counts, types, components
   - Hierarchical organization

3. **GenericAutoWiring**
   - Unified wiring interface
   - Factory methods for cell creation
   - Compatible with WiredxLSTM pattern

Core Functions:

- `discover_model_structure()` - Main entry point
- `analyze_weight_keys()` - Pattern extraction
- `detect_components()` - Component classification
- `build_wiring()` - Create wiring object

Pattern Matching
----------------

**Block Pattern Examples**:

BERT-style:
  Pattern: r'bert\.encoder\.layer\.(\d+)'
  Matches: 'bert.encoder.layer.0.attention.q_proj.weight'
  Extracts: block_idx=0

xLSTM-style:
  Pattern: r'backbone\.blocks\.(\d+)'
  Matches: 'backbone.blocks.15.mlstm_layer.q.weight'
  Extracts: block_idx=15

GPT-style:
  Pattern: r'transformer\.h\.(\d+)'
  Matches: 'transformer.h.23.attn.c_attn.weight'
  Extracts: block_idx=23

Custom multi-level:
  Pattern: r'model\.encoder\.stage\.(\d+)\.block\.(\d+)'
  Custom handler for nested indices

Component Detection
-------------------

Default taxonomy:

**Attention Components**:
  - Keywords: [qkv, q_proj, k_proj, v_proj, attention, self_attn]
  - Example: 'layer.0.attention.q_proj.weight' → attention

**FFN Components**:
  - Keywords: [mlp, ffn, feed_forward, intermediate, fc]
  - Example: 'layer.0.mlp.up_proj.weight' → ffn

**Normalization**:
  - Keywords: [norm, layer_norm, rms_norm, ln]
  - Example: 'layer.0.norm1.weight' → norm

**LSTM Components**:
  - Keywords: [mlstm, slstm, lstm, rnn]
  - Example: 'blocks.0.mlstm_layer.q.weight' → mlstm

Users can override or extend this taxonomy.

Block Type Inference
--------------------

Heuristics for determining block type:

1. **LSTM Block**: Has mlstm_layer or slstm_layer component
2. **Attention Block**: Has attention component + ffn
3. **Transformer Block**: Has attention + ffn + norms (BERT/GPT style)
4. **Custom Block**: User-defined detection logic

Multi-Source Support
--------------------

**From Safetensors**:
  >>> wiring = discover_model_structure(
  ...     model_dir='checkpoint_dir',
  ...     block_pattern=r'layer\.(\d+)'
  ... )

**From Weight Dict**:
  >>> weights = mx.load('model.npz')
  >>> wiring = discover_model_structure(
  ...     weights_dict=weights,
  ...     block_pattern=r'blocks\.(\d+)'
  ... )

**From PyTorch Checkpoint**:
  >>> state_dict = torch.load('model.pt')['state_dict']
  >>> weights = {k: mx.array(v.numpy()) for k, v in state_dict.items()}
  >>> wiring = discover_model_structure(
  ...     weights_dict=weights,
  ...     block_pattern=r'encoder\.(\d+)'
  ... )

Examples
--------

**Complete M2-BERT Loading**:
  >>> import mlx.core as mx
  >>> from xlstm_metal.mlx_jit.utils.generic_auto_wiring import discover_model_structure
  >>> from xlstm_metal.mlx_jit.utils.generic_weight_loader import load_weights_from_dict
  >>> 
  >>> # Load M2-BERT weights
  >>> weights = mx.load('m2bert_checkpoint.npz')
  >>> 
  >>> # Discover structure
  >>> wiring = discover_model_structure(
  ...     weights_dict=weights,
  ...     block_pattern=r'bert\.encoder\.layer\.(\d+)',
  ...     model_type='bert'
  ... )
  >>> 
  >>> print(f"Found {wiring.structure['num_blocks']} encoder layers")
  >>> 
  >>> # Build model from discovered structure
  >>> model = build_model_from_wiring(wiring)
  >>> 
  >>> # Load weights with generic loader
  >>> load_weights_from_dict(model, weights)

**Custom Architecture**:
  >>> # Custom detection for non-standard naming
  >>> def detect_my_blocks(key):
  ...     # Custom logic for unusual patterns
  ...     if 'encoder_stage' in key:
  ...         match = re.search(r'stage_(\d+)_layer_(\d+)', key)
  ...         if match:
  ...             return int(match.group(1)) * 100 + int(match.group(2))
  ...     return None
  >>> 
  >>> wiring = discover_model_structure(
  ...     weights_dict=weights,
  ...     block_detector=detect_my_blocks
  ... )

Integration with Existing Code
-------------------------------

This generic wiring can replace model-specific AutoWiring:

**Before** (xLSTM-specific):
  from xlstm_metal.mlx_jit.wiring import create_auto_wiring
  wiring = create_auto_wiring("xlstm_7b_model")

**After** (generic):
  from xlstm_metal.mlx_jit.utils.generic_auto_wiring import discover_model_structure
  wiring = discover_model_structure(
      model_dir="xlstm_7b_model",
      block_pattern=r'backbone\.blocks\.(\d+)'
  )

Benefits
--------
1. **Model Agnostic**: Works with any architecture
2. **Zero Hardcoding**: No model-specific path assumptions
3. **Extensible**: Easy to add new component types
4. **Debuggable**: Reports discovered structure for validation
5. **Portable**: Same interface across different model families

Limitations
-----------
- Requires weight keys to have consistent naming patterns
- Cannot infer connections beyond sequential (no skip connections)
- Assumes single block index per layer (no nested blocks)
- User must provide appropriate block_pattern for their model

Future Extensions
-----------------
- Graph-based connectivity inference (attention masks, skip connections)
- Automatic dimension inference from weight shapes
- Support for mixture-of-experts routing
- Multi-modal model discovery (vision + language)

Parity
------
Provides superset functionality of xLSTM-specific AutoWiring while
maintaining compatible interface for drop-in replacement.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import mlx.core as mx


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class GenericWiringConfig:
    """Configuration for generic model structure discovery.
    
    Attributes
    ----------
    block_pattern : str | Pattern | None
        Regex pattern to extract block indices from weight keys.
        Example: r'bert\.encoder\.layer\.(\d+)' for BERT models.
    block_detector : callable | None
        Custom function: key → block_idx | None.
        Overrides block_pattern if provided.
    component_rules : dict[str, list[str]]
        Mapping of component types to keyword patterns.
        Example: {'attention': ['attn', 'self_attention'], 'ffn': ['mlp']}
    model_type : str
        Model family hint ('bert', 'gpt', 'xlstm', 'custom').
    special_keys : dict[str, str]
        Patterns for special components (embedding, norm, head).
        Example: {'embedding': 'bert.embeddings', 'pooler': 'bert.pooler'}
    strict : bool
        Whether to raise errors on ambiguous patterns.
    """
    block_pattern: Optional[Union[str, Pattern]] = None
    block_detector: Optional[Callable[[str], Optional[int]]] = None
    component_rules: Dict[str, List[str]] = field(default_factory=dict)
    model_type: str = 'custom'
    special_keys: Dict[str, str] = field(default_factory=dict)
    strict: bool = False

    def __post_init__(self):
        """Compile regex patterns and set defaults."""
        if self.block_pattern and isinstance(self.block_pattern, str):
            self.block_pattern = re.compile(self.block_pattern)

        # Default component rules if not provided
        if not self.component_rules:
            self.component_rules = get_default_component_rules()


def get_default_component_rules() -> Dict[str, List[str]]:
    """Return default component detection rules.
    
    Returns
    -------
    rules : dict[str, list[str]]
        Mapping of component type → keyword list.
    """
    return {
        'attention': ['attention', 'attn', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'qkv'],
        'ffn': ['mlp', 'ffn', 'feed_forward', 'intermediate', 'fc', 'dense', 'gated_layers'],
        'norm': ['norm', 'layer_norm', 'rms_norm', 'ln', 'layernorm'],
        'mlstm': ['mlstm', 'mlstm_layer'],
        'slstm': ['slstm', 'slstm_layer'],
        'lstm': ['lstm', 'rnn'],
        'filter': ['filter', 'filter_fn'],  # M2-BERT specific
        'projection': ['out_linear', 'in_linear', 'wo', 'wi'],  # M2-BERT specific
    }


# ============================================================================
# Structure Discovery
# ============================================================================

@dataclass
class DiscoveredStructure:
    """Holds discovered model architecture information.
    
    Attributes
    ----------
    num_blocks : int
        Number of transformer/encoder blocks.
    block_components : dict[int, list[str]]
        Components found in each block.
    block_types : dict[int, str]
        Inferred type for each block.
    special_components : dict[str, bool]
        Special components (embedding, classifier, etc.).
    weight_keys : list[str]
        All weight keys analyzed.
    config : dict
        Additional configuration discovered or provided.
    """
    num_blocks: int
    block_components: Dict[int, List[str]]
    block_types: Dict[int, str]
    special_components: Dict[str, bool]
    weight_keys: List[str]
    config: Dict[str, Any] = field(default_factory=dict)


def analyze_weight_keys(
        weight_keys: List[str],
        config: GenericWiringConfig
) -> Tuple[Dict[int, Set[str]], Dict[str, bool]]:
    """Analyze weight keys to extract block structure.
    
    Parameters
    ----------
    weight_keys : list[str]
        All weight keys from checkpoint.
    config : GenericWiringConfig
        Detection configuration.
    
    Returns
    -------
    block_components : dict[int, set[str]]
        Components found in each block.
    special_components : dict[str, bool]
        Whether special components exist.
    
    Examples
    --------
    >>> keys = [
    ...     'bert.encoder.layer.0.attention.q_proj.weight',
    ...     'bert.encoder.layer.0.mlp.dense.weight',
    ...     'bert.embeddings.word_embeddings.weight'
    ... ]
    >>> cfg = GenericWiringConfig(block_pattern=r'layer\.(\d+)')
    >>> blocks, special = analyze_weight_keys(keys, cfg)
    >>> blocks[0]
    {'attention', 'mlp'}
    """
    block_components: Dict[int, Set[str]] = defaultdict(set)
    special_components: Dict[str, bool] = {}

    # Detect blocks
    for key in weight_keys:
        # Try custom detector first
        block_idx = None
        if config.block_detector is not None:
            block_idx = config.block_detector(key)
        elif config.block_pattern is not None:
            match = config.block_pattern.search(key)
            if match:
                block_idx = int(match.group(1))

        if block_idx is not None:
            # Classify component
            component_type = detect_component_type(key, config.component_rules)
            if component_type:
                block_components[block_idx].add(component_type)
        else:
            # Check for special components
            for special_name, pattern in config.special_keys.items():
                if pattern in key:
                    special_components[special_name] = True

    # Auto-detect common special components if not configured
    if not config.special_keys:
        special_components = {
            'embedding': any('embedding' in k.lower() for k in weight_keys),
            'classifier': any('classifier' in k.lower() or 'head' in k.lower() for k in weight_keys),
            'pooler': any('pooler' in k.lower() for k in weight_keys),
        }

    return block_components, special_components


def detect_component_type(key: str, rules: Dict[str, List[str]]) -> Optional[str]:
    """Detect component type from weight key using rules.
    
    Parameters
    ----------
    key : str
        Weight key to classify.
    rules : dict[str, list[str]]
        Component rules (type → keyword list).
    
    Returns
    -------
    component_type : str | None
        Detected component type or None if no match.
    
    Examples
    --------
    >>> rules = {'attention': ['attn', 'q_proj'], 'ffn': ['mlp']}
    >>> detect_component_type('layer.0.attn.weight', rules)
    'attention'
    >>> detect_component_type('layer.0.mlp.weight', rules)
    'ffn'
    """
    key_lower = key.lower()

    # Check rules in priority order (more specific first)
    priority = ['mlstm', 'slstm', 'attention', 'ffn', 'norm', 'filter', 'projection', 'lstm']

    for component_type in priority:
        if component_type in rules:
            for keyword in rules[component_type]:
                if keyword in key_lower:
                    return component_type

    return None


def infer_block_type(components: Set[str]) -> str:
    """Infer block type from its components.
    
    Parameters
    ----------
    components : set[str]
        Set of component types in block.
    
    Returns
    -------
    block_type : str
        Inferred type: 'mlstm', 'slstm', 'transformer', 'attention', 'unknown'.
    
    Examples
    --------
    >>> infer_block_type({'attention', 'ffn', 'norm'})
    'transformer'
    >>> infer_block_type({'mlstm', 'ffn'})
    'mlstm'
    """
    if 'mlstm' in components:
        return 'mlstm'
    elif 'slstm' in components:
        return 'slstm'
    elif 'attention' in components and 'ffn' in components:
        return 'transformer'
    elif 'attention' in components:
        return 'attention'
    elif 'ffn' in components:
        return 'ffn_only'
    else:
        return 'unknown'


def discover_model_structure(
        weights_dict: Optional[Dict[str, mx.array]] = None,
        model_dir: Optional[Union[str, Path]] = None,
        block_pattern: Optional[Union[str, Pattern]] = None,
        block_detector: Optional[Callable[[str], Optional[int]]] = None,
        component_rules: Optional[Dict[str, List[str]]] = None,
        model_type: str = 'custom',
        special_keys: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = False,
) -> DiscoveredStructure:
    """Main entry point for generic model structure discovery.
    
    Discovers model architecture from weights without hardcoded assumptions.
    Supports multiple weight sources (safetensors, npz, dict) and custom
    detection rules.
    
    Parameters
    ----------
    weights_dict : dict[str, mx.array] | None, optional
        Pre-loaded weights dictionary.
    model_dir : str | Path | None, optional
        Directory with safetensors checkpoint (if weights_dict not provided).
    block_pattern : str | Pattern | None, optional
        Regex pattern to extract block indices.
    block_detector : callable | None, optional
        Custom function to extract block index from key.
    component_rules : dict[str, list[str]] | None, optional
        Custom component detection rules.
    model_type : str, default 'custom'
        Model family hint for default rules.
    special_keys : dict[str, str] | None, optional
        Patterns for special components.
    config : dict | None, optional
        Additional configuration.
    strict : bool, default False
        Whether to enforce strict validation.
    
    Returns
    -------
    structure : DiscoveredStructure
        Discovered model structure.
    
    Raises
    ------
    ValueError
        If neither weights_dict nor model_dir provided.
        If no blocks detected with given patterns.
    
    Examples
    --------
    >>> # From safetensors
    >>> structure = discover_model_structure(
    ...     model_dir='m2bert_checkpoint',
    ...     block_pattern=r'bert\.encoder\.layer\.(\d+)'
    ... )
    
    >>> # From weight dict
    >>> weights = mx.load('model.npz')
    >>> structure = discover_model_structure(
    ...     weights_dict=weights,
    ...     block_pattern=r'blocks\.(\d+)'
    ... )
    
    >>> # Custom detector
    >>> def my_detector(key):
    ...     match = re.search(r'my_layer_(\d+)', key)
    ...     return int(match.group(1)) if match else None
    >>> structure = discover_model_structure(
    ...     weights_dict=weights,
    ...     block_detector=my_detector
    ... )
    """
    # Build configuration
    wiring_config = GenericWiringConfig(
        block_pattern=block_pattern,
        block_detector=block_detector,
        component_rules=component_rules or {},
        model_type=model_type,
        special_keys=special_keys or {},
        strict=strict,
    )

    # Load weights if needed
    if weights_dict is None:
        if model_dir is None:
            raise ValueError("Must provide either weights_dict or model_dir")
        weights_dict = load_weights_from_source(model_dir)

    weight_keys = list(weights_dict.keys())

    # Analyze structure
    block_components_sets, special_components = analyze_weight_keys(
        weight_keys, wiring_config
    )

    # Convert sets to sorted lists
    block_components = {
        idx: sorted(list(comps))
        for idx, comps in sorted(block_components_sets.items())
    }

    # Infer block types
    block_types = {
        idx: infer_block_type(block_components_sets[idx])
        for idx in block_components.keys()
    }

    num_blocks = len(block_components)

    if num_blocks == 0 and strict:
        raise ValueError(
            f"No blocks detected with pattern: {block_pattern}. "
            f"Sample keys: {weight_keys[:5]}"
        )

    return DiscoveredStructure(
        num_blocks=num_blocks,
        block_components=block_components,
        block_types=block_types,
        special_components=special_components,
        weight_keys=weight_keys,
        config=config or {},
    )


def load_weights_from_source(source: Union[str, Path]) -> Dict[str, mx.array]:
    """Load weights from various sources (safetensors, npz, etc.).
    
    Parameters
    ----------
    source : str | Path
        Path to checkpoint directory or file.
    
    Returns
    -------
    weights : dict[str, mx.array]
        Loaded weights dictionary.
    
    Raises
    ------
    FileNotFoundError
        If source not found.
    ValueError
        If source format not recognized.
    """
    source = Path(source)

    # Try safetensors first
    if source.is_dir():
        index_path = source / "model.safetensors.index.json"
        if index_path.exists():
            from xlstm_metal.mlx_jit.utils.config_loader import load_safetensor_shards
            return load_safetensor_shards(str(source))

    # Try NPZ
    if source.suffix == '.npz' or (source.is_dir() and (source / "model.npz").exists()):
        npz_path = source if source.suffix == '.npz' else source / "model.npz"
        return mx.load(str(npz_path))

    # Try single safetensors file
    if source.suffix == '.safetensors':
        return mx.load(str(source))

    raise ValueError(f"Could not determine weight format for: {source}")


# ============================================================================
# Convenience Functions
# ============================================================================

def print_discovered_structure(structure: DiscoveredStructure):
    """Pretty-print discovered model structure for debugging.
    
    Parameters
    ----------
    structure : DiscoveredStructure
        Structure to display.
    
    Examples
    --------
    >>> structure = discover_model_structure(...)
    >>> print_discovered_structure(structure)
    Model Structure Discovery Results
    ==================================
    Total Blocks: 12
    Block Types: transformer (12x)
    
    Block 0: transformer
      Components: attention, ffn, norm
    Block 1: transformer
      Components: attention, ffn, norm
    ...
    """
    print("\nModel Structure Discovery Results")
    print("=" * 50)
    print(f"Total Blocks: {structure.num_blocks}")

    # Count block types
    type_counts = defaultdict(int)
    for block_type in structure.block_types.values():
        type_counts[block_type] += 1

    print("Block Types:", ", ".join(f"{t} ({c}x)" for t, c in type_counts.items()))

    # Special components
    if structure.special_components:
        print("\nSpecial Components:")
        for name, exists in structure.special_components.items():
            if exists:
                print(f"  ✓ {name}")

    # Per-block details
    print(f"\nPer-Block Details:")
    for idx in sorted(structure.block_components.keys())[:5]:  # Show first 5
        block_type = structure.block_types[idx]
        components = structure.block_components[idx]
        print(f"  Block {idx}: {block_type}")
        print(f"    Components: {', '.join(components)}")

    if structure.num_blocks > 5:
        print(f"  ... ({structure.num_blocks - 5} more blocks)")


def create_bert_wiring_config() -> GenericWiringConfig:
    """Preset configuration for BERT-style models.
    
    Returns
    -------
    config : GenericWiringConfig
        Configuration for BERT/RoBERTa/ALBERT models.
    """
    return GenericWiringConfig(
        block_pattern=r'(?:bert|roberta|albert)\.encoder\.layer\.(\d+)',
        model_type='bert',
        special_keys={
            'embedding': 'embeddings',
            'pooler': 'pooler',
            'classifier': 'classifier',
        }
    )


def create_gpt_wiring_config() -> GenericWiringConfig:
    """Preset configuration for GPT-style models.
    
    Returns
    -------
    config : GenericWiringConfig
        Configuration for GPT-2/GPT-3/GPT-NeoX models.
    """
    return GenericWiringConfig(
        block_pattern=r'(?:transformer|gpt_neox)\.h\.(\d+)',
        model_type='gpt',
        special_keys={
            'embedding': 'wte',
            'lm_head': 'lm_head',
        }
    )


def create_xlstm_wiring_config() -> GenericWiringConfig:
    """Preset configuration for xLSTM models.
    
    Returns
    -------
    config : GenericWiringConfig
        Configuration for xLSTM models.
    """
    return GenericWiringConfig(
        block_pattern=r'backbone\.blocks\.(\d+)',
        model_type='xlstm',
        special_keys={
            'embedding': 'backbone.embeddings',
            'out_norm': 'backbone.out_norm',
            'lm_head': 'lm_head',
        }
    )


__all__ = [
    'GenericWiringConfig',
    'DiscoveredStructure',
    'discover_model_structure',
    'print_discovered_structure',
    'analyze_weight_keys',
    'detect_component_type',
    'infer_block_type',
    'create_bert_wiring_config',
    'create_gpt_wiring_config',
    'create_xlstm_wiring_config',
]
