from __future__ import annotations
"""
Profile loader for upstream xLSTM using the flexible block-stack API.

Maps a JSON dict into xLSTMBlockStackConfig and xLSTMLMModelConfig without shims.

Failure policy: raise on missing/invalid fields; no try/except wrappers.
"""

from dataclasses import asdict
from typing import Any, Dict

from .xlstm_block_stack import xLSTMBlockStackConfig
from .xlstm_lm_model import xLSTMLMModelConfig
from .blocks.mlstm.block import mLSTMBlockConfig
from .blocks.mlstm.layer import mLSTMLayerConfig
from .blocks.slstm.block import sLSTMBlockConfig
from .blocks.slstm.layer import sLSTMLayerConfig
from .components.feedforward import FeedForwardConfig


def _dict_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in d:
        return d[key]
    if default is not None:
        return default
    raise KeyError(f"Missing required key: {key}")


def build_block_stack_config(profile: Dict[str, Any]) -> xLSTMBlockStackConfig:
    """Construct xLSTMBlockStackConfig from a plain dict.

    Expected keys (minimal):
      - embedding_dim: int
      - num_blocks: int
      - context_length: int
      - slstm_at: list[int] or "all"

    Optional sub-configs:
      - mlstm: { conv1d_kernel_size, qkv_proj_blocksize, num_heads, proj_factor, bias, dropout }
      - slstm: { embedding_dim, num_heads, ... } (as defined by sLSTMLayerConfig)
      - feedforward: { proj_factor, act_fn, dropout, bias, ff_type }
    """
    emb = int(_dict_get(profile, "embedding_dim"))
    num_blocks = int(_dict_get(profile, "num_blocks"))
    ctx = int(profile.get("context_length", -1))

    # Build mLSTM layer defaults and override from profile
    mlstm_cfg_dict: Dict[str, Any] = profile.get("mlstm", {})
    mlstm_layer = mLSTMLayerConfig(
        embedding_dim=emb,
        num_heads=int(mlstm_cfg_dict.get("num_heads", 4)),
        bias=bool(mlstm_cfg_dict.get("bias", False)),
        dropout=float(mlstm_cfg_dict.get("dropout", 0.0)),
        context_length=ctx,
        conv1d_kernel_size=int(mlstm_cfg_dict.get("conv1d_kernel_size", 4)),
        qkv_proj_blocksize=int(mlstm_cfg_dict.get("qkv_proj_blocksize", 4)),
        proj_factor=float(mlstm_cfg_dict.get("proj_factor", 2.0)),
    )
    mlstm_block = mLSTMBlockConfig(mlstm=mlstm_layer, _num_blocks=num_blocks)

    # Optional sLSTM + optional feedforward
    if "slstm" in profile:
        slstm_cfg_dict: Dict[str, Any] = profile["slstm"]
        slstm_layer = sLSTMLayerConfig(
            embedding_dim=emb,
            num_heads=int(slstm_cfg_dict.get("num_heads", 4)),
            bias=bool(slstm_cfg_dict.get("bias", False)),
            dropout=float(slstm_cfg_dict.get("dropout", 0.0)),
            context_length=ctx,
        )
        ff_cfg_dict: Dict[str, Any] = profile.get("feedforward", {})
        feedforward = FeedForwardConfig(
            embedding_dim=emb,
            proj_factor=float(ff_cfg_dict.get("proj_factor", 1.3)),
            act_fn=str(ff_cfg_dict.get("act_fn", "gelu")),
            dropout=float(ff_cfg_dict.get("dropout", 0.0)),
            bias=bool(ff_cfg_dict.get("bias", False)),
        )
        slstm_block = sLSTMBlockConfig(slstm=slstm_layer, feedforward=feedforward, _num_blocks=num_blocks)
    else:
        slstm_block = None

    slstm_at = profile.get("slstm_at", [])
    if slstm_at == "all":
        slstm_at_list = list(range(num_blocks))
    elif isinstance(slstm_at, list):
        slstm_at_list = [int(i) for i in slstm_at]
    else:
        slstm_at_list = []

    return xLSTMBlockStackConfig(
        mlstm_block=mlstm_block,
        slstm_block=slstm_block,
        context_length=ctx,
        num_blocks=num_blocks,
        embedding_dim=emb,
        add_post_blocks_norm=bool(profile.get("add_post_blocks_norm", True)),
        bias=bool(profile.get("bias", False)),
        dropout=float(profile.get("dropout", 0.0)),
        slstm_at=slstm_at_list,
    )


def build_lm_config(profile: Dict[str, Any]) -> xLSTMLMModelConfig:
    bs_cfg = build_block_stack_config(profile)
    vocab = int(_dict_get(profile, "vocab_size"))
    return xLSTMLMModelConfig(
        **asdict(bs_cfg),
        vocab_size=vocab,
        tie_weights=bool(profile.get("tie_weights", False)),
        weight_decay_on_embedding=bool(profile.get("weight_decay_on_embedding", False)),
        add_embedding_dropout=bool(profile.get("add_embedding_dropout", False)),
    )

