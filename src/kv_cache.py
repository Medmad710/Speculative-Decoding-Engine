"""
kv_cache.py — DynamicCache Utilities
=====================================
Provides safe, version-stable helpers for slicing, rolling back, and
asserting invariants on ``transformers.cache_utils.DynamicCache`` objects.

Design principle
----------------
We never touch private attributes directly (``_seen_tokens``, ``key_cache``,
etc.) where a public API exists.  The public API surface we rely on:
  - ``DynamicCache.get_seq_length()``  → current sequence length in cache
  - ``DynamicCache.crop(max_length)``  → trim KV tensors to ``max_length``
  - ``DynamicCache.key_cache / value_cache`` → list[Tensor] per layer

KV-Cache Invariant
------------------
At the START of every speculative step, with a confirmed prefix of length N:
  - Both draft and target ``DynamicCache`` objects have ``seq_len = N - 1``
  - ``current_token`` tensor is the N-th token (shape ``[1, 1]``)
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_cache_seq_len(cache: DynamicCache) -> int:
    """Return the current sequence length stored in the cache.

    Uses the public ``get_seq_length()`` method.
    Compatible with both transformers 4.x and 5.x.
    """
    return int(cache.get_seq_length())


def crop_cache(cache: DynamicCache, target_len: int) -> None:
    """Trim the KV-cache to ``target_len`` tokens in-place.

    Handles compatibility across all transformers versions (4.x, 5.x, and nightly builds).
    Works whether layer.crop expects absolute length or negative tokens_to_remove.

    Args:
        cache:      The ``DynamicCache`` to trim.
        target_len: Desired sequence length after cropping.

    Raises:
        ValueError: If ``target_len`` is negative or larger than current len.
    """
    current = get_cache_seq_len(cache)
    if target_len < 0:
        raise ValueError(f"target_len must be >= 0, got {target_len}")
    if target_len > current:
        raise ValueError(
            f"Cannot crop cache from {current} to {target_len}: "
            "target_len exceeds current length."
        )
    if target_len == current:
        return

    diff = current - target_len

    # In modern transformers (5.x+), cache.layers is present.
    # Some versions require negative tokens_to_remove (-diff), some require target_len.
    if hasattr(cache, "layers") and cache.layers:
        # Try cache-level crop with negative offset first, then target_len
        for val in [-diff, target_len]:
            try:
                cache.crop(val)
                if get_cache_seq_len(cache) == target_len:
                    return
            except Exception:
                pass

        # Per-layer fallback
        for layer in cache.layers:
            cropped = False
            for val in [-diff, target_len]:
                try:
                    layer.crop(val)
                    if hasattr(layer, "get_seq_length") and layer.get_seq_length() == target_len:
                        cropped = True
                        break
                except Exception:
                    pass
            if not cropped:
                if hasattr(layer, "keys") and layer.keys is not None:
                    layer.keys = layer.keys[..., :target_len, :]
                if hasattr(layer, "values") and layer.values is not None:
                    layer.values = layer.values[..., :target_len, :]
        return

    # Transformers 4.x fallback
    try:
        cache.crop(target_len)
        if get_cache_seq_len(cache) == target_len:
            return
    except Exception:
        pass

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None:
                cache.key_cache[i] = cache.key_cache[i][..., :target_len, :]
            if cache.value_cache[i] is not None:
                cache.value_cache[i] = cache.value_cache[i][..., :target_len, :]
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = target_len


def assert_cache_invariant(
    cache: DynamicCache,
    expected_len: int,
    label: str = "cache",
) -> None:
    """Hard-assert that the cache holds exactly ``expected_len`` tokens.

    Args:
        cache:        Cache to inspect.
        expected_len: Expected ``get_seq_length()`` value.
        label:        Human-readable name for error messages.

    Raises:
        AssertionError: On mismatch.
    """
    actual = get_cache_seq_len(cache)
    assert actual == expected_len, (
        f"[{label}] Cache invariant violated: "
        f"expected seq_len={expected_len}, got {actual}."
    )


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-copy all KV tensors into a new ``DynamicCache``.

    Works across transformers 4.x (key_cache list) and 5.x (DynamicLayer objects).

    Args:
        cache: Source cache (not modified).

    Returns:
        A new ``DynamicCache`` with cloned key/value tensors.
    """
    try:
        return copy.deepcopy(cache)
    except Exception:
        # Fallback for older transformers if deepcopy fails
        new_cache = DynamicCache()
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            new_cache.key_cache = [t.clone() for t in cache.key_cache]
            new_cache.value_cache = [t.clone() for t in cache.value_cache]
            if hasattr(cache, "_seen_tokens"):
                new_cache._seen_tokens = cache._seen_tokens
        return new_cache


def make_cache_position(start: int, length: int, device: torch.device) -> torch.Tensor:
    """Build a ``cache_position`` tensor ``[start, start+1, ..., start+length-1]``.

    This is passed to every model forward call so that RoPE embeddings use
    the correct absolute positions, independent of the KV-cache state.

    Args:
        start:  First absolute position (== current seq length in cache).
        length: Number of new tokens being processed.
        device: Target device.

    Returns:
        1-D ``LongTensor`` of shape ``(length,)``.
    """
    return torch.arange(start, start + length, dtype=torch.long, device=device)
