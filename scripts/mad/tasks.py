from __future__ import annotations

"""
Minimal MAD-style synthetic tasks for quick capability checks.

Implements a subset of tasks (in-context recall, selective copying) without
external dependencies. Fail-fast: no try/except, assertions guard invariants.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch


IGNORE_INDEX = -100


@dataclass
class TaskConfig:
    vocab_size: int = 16
    seq_len: int = 128
    batch_size: int = 8
    multi_query: bool = False
    noise_vocab_size: int = 0
    frac_noise: float = 0.0


def generate_in_context_recall(cfg: TaskConfig, *, train: bool, rng: np.random.Generator | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = np.random.default_rng()

    B, S, V = cfg.batch_size, cfg.seq_len, cfg.vocab_size
    inputs = []
    targets = []

    for _ in range(B):
        x, y = _generate_in_context_recall_instance(
            vocab_size=V,
            seq_len=S,
            is_training=train,
            rng=rng,
            target_ignore_idx=IGNORE_INDEX,
            multi_query=cfg.multi_query,
            noise_vocab_size=cfg.noise_vocab_size,
            frac_noise=cfg.frac_noise,
        )
        inputs.append(x)
        targets.append(y)

    x_t = torch.tensor(np.stack(inputs, axis=0), dtype=torch.long)
    y_t = torch.tensor(np.stack(targets, axis=0), dtype=torch.long)
    return x_t, y_t


def generate_selective_copying(
    cfg: TaskConfig,
    *,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selective copying batch generator.

    Inputs contain tokens, blanks, and a copy token; targets contain ignore_index except
    for the final copied segment, which holds the tokens to copy in order.
    """
    if rng is None:
        rng = np.random.default_rng()
    B, S, V = cfg.batch_size, cfg.seq_len, cfg.vocab_size
    assert S > (num_tokens_to_copy * 2) + 1, "seq_len must be > (num_tokens_to_copy * 2) + 1"

    copy_token = V - 1
    blank_token = V - 2
    non_special = V - 2
    vocab = np.arange(non_special)

    xs = []
    ys = []
    for _ in range(B):
        num_blank_tokens = S - (num_tokens_to_copy * 2) - 1
        to_copy = rng.choice(vocab, size=(num_tokens_to_copy,), replace=True).reshape(-1)
        # Insert blanks randomly among the first segment
        arr = np.array(to_copy)
        insert_indices = np.random.randint(0, len(arr), num_blank_tokens)
        arr = np.insert(arr, insert_indices, [blank_token] * num_blank_tokens).tolist()
        arr += [copy_token]
        arr += [blank_token] * num_tokens_to_copy
        inputs = np.array(arr)
        targets = [IGNORE_INDEX] * (num_tokens_to_copy + num_blank_tokens + 1)
        targets += list(to_copy)
        targets = np.array(targets)
        xs.append(inputs)
        ys.append(targets)

    x_t = torch.tensor(np.stack(xs, axis=0), dtype=torch.long)
    y_t = torch.tensor(np.stack(ys, axis=0), dtype=torch.long)
    return x_t, y_t


def _generate_in_context_recall_instance(
    *,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
    rng: np.random.Generator,
    target_ignore_idx: int,
    multi_query: bool,
    noise_vocab_size: int,
    frac_noise: float,
):
    assert seq_len % 2 == 0, "seq_len must be even"
    assert 0 <= frac_noise < 1

    copy_prefix = vocab_size - 1
    non_special = vocab_size - (0 if multi_query else 1)
    non_special -= noise_vocab_size
    key_vocab = np.arange(non_special // 2)
    value_vocab = np.arange(non_special // 2, non_special)
    if frac_noise > 0:
        assert noise_vocab_size > 0
        noise_vocab = np.arange(non_special, non_special + noise_vocab_size)

    kv_map: dict[int, int] = {}
    inputs: list[int] = []
    targets: list[int] = []
    keys_presented: dict[int, int] = {}
    kv_motif = 2
    num_pairs = seq_len // kv_motif
    not_noise_idx = rng.choice(num_pairs - 1)
    for i in range(num_pairs - 1):
        is_noise = (rng.random() < frac_noise) and (i != not_noise_idx) and (frac_noise > 0)
        if is_noise:
            noise = rng.choice(noise_vocab, size=kv_motif, replace=True)
            inputs += list(noise)
            targets += [target_ignore_idx] * kv_motif
        else:
            k = int(rng.choice(key_vocab))
            if k not in kv_map:
                v = int(rng.choice(value_vocab))
                kv_map[k] = v
            else:
                v = kv_map[k]
            inputs.append(k)
            inputs.append(v)
            targets.append(target_ignore_idx)
            if k not in keys_presented:
                targets.append(target_ignore_idx)
            else:
                targets.append(v if multi_query else target_ignore_idx)
            keys_presented[k] = v

    k_probe = int(rng.choice(list(keys_presented.keys())))
    v_probe = keys_presented[k_probe]
    if not multi_query:
        inputs.append(copy_prefix)
    inputs.append(k_probe)
    inputs.append(v_probe)
    if not multi_query:
        targets.append(target_ignore_idx)
        targets.append(target_ignore_idx)
        targets.append(v_probe)
    else:
        targets.append(target_ignore_idx)
        targets.append(v_probe)

    x = np.array(inputs, dtype=int)[:-1]
    t = (x[1:] if is_training else np.array(targets[1:], dtype=int))
    return x[:-0], t


def accuracy_ignore(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> float:
    assert pred.shape == target.shape
    mask = target != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(1, total)
