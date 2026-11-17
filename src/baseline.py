"""
baseline.py — Manual Autoregressive Decoder with KV-Cache
==========================================================
Implements token-by-token greedy / stochastic decoding using
``transformers.cache_utils.DynamicCache`` and explicit ``cache_position``
passing on every forward call.

This serves as the ground-truth reference for correctness verification
in ``tests/test_equivalence.py`` and as the speed baseline in benchmarks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from .kv_cache import get_cache_seq_len, make_cache_position
from .metrics import BenchmarkResult, LatencyTimer, reset_peak_vram, get_peak_vram_mb


class BaselineDecoder:
    """Token-by-token autoregressive decoder with manual DynamicCache management.

    Args:
        model:     A HuggingFace causal-LM (already on device, eval mode).
        tokenizer: Matching tokenizer (for EOS detection).
        device:    Target device (inferred from model if not given).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        # Collect all EOS token ids (handles both eos_token_id and <|im_end|>)
        self.eos_ids: set[int] = self._collect_eos_ids()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_eos_ids(self) -> set[int]:
        """Build the set of token IDs that signal end-of-sequence."""
        eos_set: set[int] = set()
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            if isinstance(eos, int):
                eos_set.add(eos)
            else:
                eos_set.update(eos)
        # Qwen-style <|im_end|>
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != self.tokenizer.unk_token_id:
            eos_set.add(im_end)
        return eos_set

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """Sample or argmax a single next token from logits of shape ``(vocab,)``."""
        if temperature == 0.0:
            return int(logits.argmax())

        logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            # Shift cumsum right so the first token that crosses p is included
            cumprobs = torch.roll(cumprobs, 1)
            cumprobs[0] = 0.0
            mask = cumprobs > top_p
            sorted_logits[mask] = float("-inf")
            # Scatter back
            logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=generator))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        track_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BenchmarkResult]]:
        """Run autoregressive generation.

        Args:
            input_ids:      Prompt token IDs, shape ``(1, prompt_len)``.
            max_new_tokens: Maximum new tokens to generate.
            temperature:    Sampling temperature (0 = greedy).
            top_p:          Nucleus sampling threshold.
            seed:           RNG seed for reproducibility.
            track_metrics:  If True, return a populated ``BenchmarkResult``.

        Returns:
            ``(output_ids, result)`` where ``output_ids`` has shape
            ``(1, prompt_len + generated_len)`` and ``result`` is a
            ``BenchmarkResult`` or ``None``.
        """
        input_ids = input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        # Deterministic generator for stochastic sampling
        generator: Optional[torch.Generator] = None
        if temperature > 0.0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Reset VRAM stats
        reset_peak_vram(self.device)

        # ----- Prefill: process the full prompt in one pass -----
        cache = DynamicCache()
        cache_pos = make_cache_position(0, prompt_len, self.device)

        ttft_timer = LatencyTimer(self.device)
        total_timer = LatencyTimer(self.device)

        with total_timer:
            with ttft_timer:
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
            # cache now has seq_len = prompt_len
            last_logits = out.logits[0, -1, :]  # (vocab,)
            current_token = self._sample_token(last_logits, temperature, top_p, generator)

            generated: list[int] = [current_token]

            if current_token in self.eos_ids:
                # EOS on first token
                output_ids = torch.cat(
                    [input_ids, torch.tensor([[current_token]], device=self.device)], dim=1
                )
                result = self._make_result(
                    track_metrics, ttft_timer, total_timer,
                    prompt_len, generated
                )
                return output_ids, result

            # ----- Decode: one token at a time -----
            current_input = torch.tensor([[current_token]], device=self.device)

            for _ in range(max_new_tokens - 1):
                seq_len_in_cache = get_cache_seq_len(cache)
                cache_pos = make_cache_position(seq_len_in_cache, 1, self.device)

                out = self.model(
                    input_ids=current_input,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_pos,
                )
                last_logits = out.logits[0, -1, :]
                next_token = self._sample_token(last_logits, temperature, top_p, generator)
                generated.append(next_token)

                if next_token in self.eos_ids:
                    break

                current_input = torch.tensor([[next_token]], device=self.device)

        output_ids = torch.cat(
            [input_ids, torch.tensor([generated], device=self.device)], dim=1
        )
        result = self._make_result(
            track_metrics, ttft_timer, total_timer, prompt_len, generated
        )
        return output_ids, result

    def _make_result(
        self,
        track_metrics: bool,
        ttft_timer: LatencyTimer,
        total_timer: LatencyTimer,
        prompt_len: int,
        generated: list[int],
    ) -> Optional[BenchmarkResult]:
        if not track_metrics:
            return None
        n_gen = len(generated)
        elapsed = total_timer.elapsed_s
        return BenchmarkResult(
            throughput_tps=n_gen / elapsed if elapsed > 0 else 0.0,
            speedup_ratio=1.0,
            ttft_ms=ttft_timer.elapsed_ms,
            elapsed_s=elapsed,
            total_tokens=n_gen,
            prompt_tokens=prompt_len,
            peak_vram_mb=get_peak_vram_mb(self.device),
            decoder_type="baseline",
        )
