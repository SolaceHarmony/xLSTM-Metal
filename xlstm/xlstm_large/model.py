"""
Canonical Large model API backed by the upstream hybrid xLSTM block stack.

This module provides `xLSTMLarge` and `xLSTMLargeConfig` by delegating to the
Language Model wrapper in `xlstm_lm_model.py`. This aligns the Large model with
the upstream design (mLSTM/sLSTM striping, LayerNorm, gated FFN policy) instead
of the legacy per-block FFN variant.
"""

from .xlstm_lm_model import xLSTMLMModel as xLSTMLarge
from .xlstm_lm_model import xLSTMLMModelConfig as xLSTMLargeConfig
