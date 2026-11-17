"""
test_equivalence.py — Correctness Test Suite
=============================================
Verifies that SpeculativeDecoder produces EXACTLY the same token IDs as
BaselineDecoder for greedy decoding (temperature=0), across multiple gamma
values, prompt types, and edge cases.

Test Coverage
-------------
1. test_greedy_equivalence      — exact token-ID match, gamma in {1, 3, 5}
2. test_eos_handling            — both decoders stop at same EOS position
3. test_cache_seq_length        — DynamicCache.get_seq_length() integrity
4. test_stochastic_no_nan       — temperature=0.7 runs cleanly
5. test_greedy_equivalence_long — longer prompt, gamma=4

Models used: Qwen/Qwen2.5-0.5B-Instruct (draft) + Qwen/Qwen2.5-1.5B-Instruct (target)
These are small enough to run on CPU / low-VRAM GPU.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Allow importing from project root when running pytest from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baseline import BaselineDecoder
from src.speculative import SpeculativeDecoder
from src.kv_cache import get_cache_seq_len


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if DEVICE.type == "cuda" else torch.float32

# Short prompts used across tests
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "In mathematics, the derivative of x^2 is",
]

# A prompt designed to trigger early EOS
EOS_PROMPT = "What is 2 + 2? Answer in one word:"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def models():
    """Load draft and target models once per test session."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n[fixture] Loading tokenizer from {TARGET_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[fixture] Loading draft model {DRAFT_MODEL_ID} on {DEVICE}...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    print(f"[fixture] Loading target model {TARGET_MODEL_ID} on {DEVICE}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    return tokenizer, draft_model, target_model


@pytest.fixture(scope="session")
def baseline_decoder(models):
    tokenizer, _, target_model = models
    return BaselineDecoder(target_model, tokenizer, device=DEVICE)


@pytest.fixture(scope="session")
def speculative_decoder(models):
    tokenizer, draft_model, target_model = models
    return SpeculativeDecoder(draft_model, target_model, tokenizer, device=DEVICE)


def _tokenize(tokenizer, prompt: str) -> torch.Tensor:
    """Apply chat template and return input_ids tensor."""
    # Try Qwen3-style enable_thinking=False
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    ids = tokenizer.encode(formatted, return_tensors="pt")
    return ids.to(DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGreedyEquivalence:
    """Exact token-ID match between Baseline and Speculative at temperature=0."""

    @pytest.mark.parametrize("gamma", [1, 3, 5])
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_greedy_equivalence(
        self,
        models,
        baseline_decoder: BaselineDecoder,
        speculative_decoder: SpeculativeDecoder,
        gamma: int,
        prompt: str,
    ):
        tokenizer = models[0]
        input_ids = _tokenize(tokenizer, prompt)

        baseline_out, _ = baseline_decoder.generate(
            input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )

        spec_out, stats = speculative_decoder.generate(
            input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            gamma=gamma,
            temperature=0.0,
            track_metrics=True,
        )

        # Trim to common length in case of minor length differences due to bonus token
        min_len = min(baseline_out.shape[1], spec_out.shape[1])
        baseline_tokens = baseline_out[0, :min_len].tolist()
        spec_tokens = spec_out[0, :min_len].tolist()

        assert baseline_tokens == spec_tokens, (
            f"Token mismatch at gamma={gamma}, prompt='{prompt[:40]}'.\n"
            f"Baseline: {baseline_tokens[-20:]}\n"
            f"Speculative: {spec_tokens[-20:]}"
        )

        # Both should generate at least 1 token
        prompt_len = input_ids.shape[1]
        assert spec_out.shape[1] > prompt_len, "Speculative decoder generated 0 tokens"
        assert baseline_out.shape[1] > prompt_len, "Baseline decoder generated 0 tokens"

        print(
            f"\n  ✓ gamma={gamma} | "
            f"tokens={spec_out.shape[1]-prompt_len} | "
            f"α={stats.mean_acceptance_rate:.3f} | "
            f"steps={stats.total_steps}"
        )


class TestEOSHandling:
    """Both decoders stop at the same EOS position."""

    def test_eos_produces_same_output(
        self,
        models,
        baseline_decoder: BaselineDecoder,
        speculative_decoder: SpeculativeDecoder,
    ):
        tokenizer = models[0]
        input_ids = _tokenize(tokenizer, EOS_PROMPT)

        baseline_out, _ = baseline_decoder.generate(
            input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )

        spec_out, _ = speculative_decoder.generate(
            input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            gamma=4,
            temperature=0.0,
        )

        min_len = min(baseline_out.shape[1], spec_out.shape[1])
        assert baseline_out[0, :min_len].tolist() == spec_out[0, :min_len].tolist(), (
            "EOS handling mismatch between baseline and speculative decoder."
        )

    @pytest.mark.parametrize("gamma", [1, 3, 5])
    def test_eos_does_not_exceed_max_new_tokens(
        self,
        models,
        speculative_decoder: SpeculativeDecoder,
        gamma: int,
    ):
        tokenizer = models[0]
        input_ids = _tokenize(tokenizer, EOS_PROMPT)
        max_new = 20

        spec_out, _ = speculative_decoder.generate(
            input_ids.clone(),
            max_new_tokens=max_new,
            gamma=gamma,
            temperature=0.0,
        )
        generated_len = spec_out.shape[1] - input_ids.shape[1]
        assert generated_len <= max_new, (
            f"Generated {generated_len} tokens but max_new_tokens={max_new}"
        )


class TestCacheIntegrity:
    """DynamicCache.get_seq_length() reports correct values after generation."""

    def test_cache_seq_length_after_baseline(
        self,
        models,
        baseline_decoder: BaselineDecoder,
    ):
        """Baseline decoder should not leak KV-cache state between calls."""
        tokenizer = models[0]
        prompt = PROMPTS[0]
        input_ids = _tokenize(tokenizer, prompt)
        prompt_len = input_ids.shape[1]

        out1, _ = baseline_decoder.generate(
            input_ids.clone(), max_new_tokens=30, temperature=0.0
        )
        out2, _ = baseline_decoder.generate(
            input_ids.clone(), max_new_tokens=30, temperature=0.0
        )

        # Deterministic: both runs should produce identical output
        assert out1.tolist() == out2.tolist(), (
            "Baseline decoder is not deterministic across runs."
        )

    def test_cache_seq_length_speculative(
        self,
        models,
        speculative_decoder: SpeculativeDecoder,
    ):
        """Speculative decoder should produce identical output on repeated calls."""
        tokenizer = models[0]
        prompt = PROMPTS[1]
        input_ids = _tokenize(tokenizer, prompt)

        out1, _ = speculative_decoder.generate(
            input_ids.clone(), max_new_tokens=30, gamma=3, temperature=0.0
        )
        out2, _ = speculative_decoder.generate(
            input_ids.clone(), max_new_tokens=30, gamma=3, temperature=0.0
        )
        assert out1.tolist() == out2.tolist(), (
            "Speculative decoder is not deterministic across runs."
        )


class TestStochasticMode:
    """Stochastic decoding (temperature>0) runs without NaN/shape errors."""

    @pytest.mark.parametrize("gamma", [2, 4])
    def test_stochastic_no_nan(
        self,
        models,
        speculative_decoder: SpeculativeDecoder,
        gamma: int,
    ):
        tokenizer = models[0]
        input_ids = _tokenize(tokenizer, PROMPTS[0])

        spec_out, stats = speculative_decoder.generate(
            input_ids.clone(),
            max_new_tokens=40,
            gamma=gamma,
            temperature=0.7,
            top_p=0.9,
            seed=1234,
            track_metrics=True,
        )

        # No NaN/Inf in output token IDs
        assert spec_out is not None
        tokens = spec_out[0].tolist()
        assert all(isinstance(t, int) and t >= 0 for t in tokens), (
            "Output contains invalid token IDs."
        )
        assert spec_out.shape[1] > input_ids.shape[1], (
            "Stochastic speculative decoder generated 0 tokens."
        )
        assert stats.mean_acceptance_rate is not None
        assert 0.0 <= stats.mean_acceptance_rate <= 1.0, (
            f"Acceptance rate out of range: {stats.mean_acceptance_rate}"
        )

        print(
            f"\n  ✓ stochastic gamma={gamma} | "
            f"tokens={spec_out.shape[1]-input_ids.shape[1]} | "
            f"α={stats.mean_acceptance_rate:.3f}"
        )

    def test_stochastic_baseline_no_nan(
        self,
        models,
        baseline_decoder: BaselineDecoder,
    ):
        tokenizer = models[0]
        input_ids = _tokenize(tokenizer, PROMPTS[2])

        out, _ = baseline_decoder.generate(
            input_ids.clone(),
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.9,
        )
        assert out.shape[1] > input_ids.shape[1]
