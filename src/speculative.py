"""
speculative.py — Speculative Decoding Engine
=============================================
Implements the exact lossless speculative decoding algorithm from:
  - Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
  - Chen et al. (2023) "Accelerating Large Language Model Decoding with Speculative Sampling"

Mathematical Formulation & Algorithm
-------------------------------------
Confirmed prefix: t_1, ..., t_N
Current token:    t_N
KV-cache state:   Both draft and target caches have seq_len = N - 1

1. DRAFT PHASE:
   Draft model generates γ tokens [d_1, ..., d_γ] autoregressively.
   Inputs processed by draft model: t_N, d_1, ..., d_{γ-1}.
   Draft cache seq_len becomes (N - 1) + γ = N + γ - 1.

2. VERIFY PHASE:
   Target model processes [t_N, d_1, ..., d_γ] in ONE parallel forward pass
   of length γ + 1, starting at cache position N - 1.
   Produces γ + 1 next-token distributions:
     p_t[0]       verifies d_1  (given prefix up to t_N)
     p_t[1]       verifies d_2  (given prefix up to d_1)
     ...
     p_t[γ-1]     verifies d_γ  (given prefix up to d_{γ-1})
     p_t[γ]       predicts bonus token (given prefix up to d_γ)
   Target cache seq_len becomes (N - 1) + (γ + 1) = N + γ.

3. REJECTION SAMPLING:
   For i = 0 to γ - 1:
     - Greedy (temp=0):
         target_tok = argmax(p_t[i])
         if target_tok == d_{i+1}: accept d_{i+1}
         else: emit target_tok as correction token; stop verification (break)
     - Stochastic (temp>0):
         accept d_{i+1} with probability min(1, p_t(d_{i+1}) / p_d(d_{i+1}))
         if rejected: sample correction token from normalized residual
                      p'(x) = norm(max(0, p_t(x) - p_d(x))); stop (break)

4. BONUS TOKEN:
   If all γ tokens accepted (k == γ):
     Sample bonus token b from p_t[γ].
     Emit b.

   Notice: EVERY speculative step emits at least 1 token (k + 1 tokens).

5. KV-CACHE ROLLBACK & ADVANCE:
   - If k < γ (rejection at position k):
       Emitted: d_1, ..., d_k, plus correction token c.
       Confirmed prefix now includes t_N, d_1, ..., d_k.
       Inputs processed into cache: t_N, d_1, ..., d_k (length N + k).
       Target cache is cropped to N + k.
       Draft cache (which processed up to d_k as input for drafting d_{k+1})
       is cropped to N + k.
       Next token to process: c.
   - If k == γ (all γ draft tokens accepted):
       Emitted: d_1, ..., d_γ, plus bonus token b.
       Target cache already processed t_N, d_1, ..., d_γ (length N + γ).
       Draft cache processed up to d_{γ-1} (length N + γ - 1).
       Feed d_γ into draft model (1 forward pass) to advance draft cache to N + γ.
       Next token to process: b.

   Invariant holds: Both caches have seq_len = N_new - 1.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from .kv_cache import (
    crop_cache,
    get_cache_seq_len,
    make_cache_position,
)
from .metrics import BenchmarkResult, LatencyTimer, get_peak_vram_mb, reset_peak_vram


class SpeculativeDecoder:
    """Lossless speculative decoding engine with exact rejection sampling.

    Args:
        draft_model:    Small draft model (eval mode).
        target_model:   Large target model (eval mode).
        tokenizer:      Shared tokenizer.
        device:         Target device (inferred from target_model if not given).
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device or next(target_model.parameters()).device
        self.eos_ids: Set[int] = self._collect_eos_ids()

    def _collect_eos_ids(self) -> Set[int]:
        eos_set: Set[int] = set()
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            if isinstance(eos, int):
                eos_set.add(eos)
            else:
                eos_set.update(eos)
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != self.tokenizer.unk_token_id:
            eos_set.add(im_end)
        return eos_set

    @staticmethod
    def _probs_from_logits(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature == 0.0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
            return probs

        logits = logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs_sorted = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs_sorted, dim=-1)
            cumprobs_shifted = torch.roll(cumprobs, 1)
            cumprobs_shifted[0] = 0.0
            remove_mask = cumprobs_shifted > top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

        return F.softmax(logits, dim=-1)

    @staticmethod
    def _sample_from_probs(
        probs: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> int:
        return int(torch.multinomial(probs, num_samples=1, generator=generator))

    @staticmethod
    def _adjusted_residual(
        p_target: torch.Tensor,
        p_draft: torch.Tensor,
    ) -> torch.Tensor:
        residual = torch.clamp(p_target - p_draft, min=0.0)
        total = residual.sum()
        if total < 1e-8:
            return p_target
        return residual / total

    @torch.inference_mode()
    def _draft_tokens(
        self,
        current_token: torch.Tensor,
        draft_cache: DynamicCache,
        gamma: int,
        temperature: float,
        top_p: float,
        generator: Optional[torch.Generator],
    ) -> Tuple[List[int], List[torch.Tensor]]:
        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []

        inp = current_token.to(self.device)

        for _ in range(gamma):
            seq_in_cache = get_cache_seq_len(draft_cache)
            cache_pos = make_cache_position(seq_in_cache, 1, self.device)

            out = self.draft_model(
                input_ids=inp,
                past_key_values=draft_cache,
                use_cache=True,
                cache_position=cache_pos,
            )
            logits = out.logits[0, -1, :]
            probs = self._probs_from_logits(logits, temperature, top_p)

            if temperature == 0.0:
                token = int(probs.argmax())
            else:
                token = self._sample_from_probs(probs, generator)

            draft_tokens.append(token)
            draft_probs.append(probs)
            inp = torch.tensor([[token]], device=self.device)

        return draft_tokens, draft_probs

    @torch.inference_mode()
    def _verify_tokens(
        self,
        current_token: torch.Tensor,
        draft_tokens: List[int],
        target_cache: DynamicCache,
        temperature: float,
        top_p: float,
    ) -> List[torch.Tensor]:
        all_tokens = [current_token.item()] + draft_tokens
        input_ids = torch.tensor([all_tokens], device=self.device)

        seq_in_cache = get_cache_seq_len(target_cache)
        gamma_plus_1 = len(all_tokens)
        cache_pos = make_cache_position(seq_in_cache, gamma_plus_1, self.device)

        out = self.target_model(
            input_ids=input_ids,
            past_key_values=target_cache,
            use_cache=True,
            cache_position=cache_pos,
        )
        logits = out.logits[0]

        target_probs: List[torch.Tensor] = []
        for i in range(logits.shape[0]):
            p = self._probs_from_logits(logits[i], temperature, top_p)
            target_probs.append(p)

        return target_probs

    @torch.inference_mode()
    def _prefill(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[DynamicCache, DynamicCache, torch.Tensor]:
        prompt_len = input_ids.shape[1]

        target_cache = DynamicCache()
        cache_pos = make_cache_position(0, prompt_len, self.device)
        out_t = self.target_model(
            input_ids=input_ids,
            past_key_values=target_cache,
            use_cache=True,
            cache_position=cache_pos,
        )

        draft_cache = DynamicCache()
        _ = self.draft_model(
            input_ids=input_ids,
            past_key_values=draft_cache,
            use_cache=True,
            cache_position=cache_pos,
        )

        return draft_cache, target_cache, out_t.logits[0, -1, :]

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        gamma: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        track_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BenchmarkResult]]:
        """Run speculative decoding.

        Args:
            input_ids:      Prompt token IDs, shape (1, prompt_len).
            max_new_tokens: Maximum new tokens to generate.
            gamma:          Draft window size.
            temperature:    Sampling temperature (0.0 = greedy).
            top_p:          Nucleus sampling threshold.
            seed:           RNG seed.
            track_metrics:  Return a populated BenchmarkResult.

        Returns:
            (output_ids, result)
        """
        input_ids = input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        generator: Optional[torch.Generator] = None
        if temperature > 0.0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        reset_peak_vram(self.device)

        ttft_timer = LatencyTimer(self.device)
        total_timer = LatencyTimer(self.device)

        total_accepted_draft = 0
        total_proposed_draft = 0
        total_steps = 0

        with total_timer:
            # Prefill phase
            with ttft_timer:
                draft_cache, target_cache, prefill_logits = self._prefill(input_ids)

            if temperature == 0.0:
                first_token = int(prefill_logits.argmax())
            else:
                first_probs = self._probs_from_logits(prefill_logits, temperature, top_p)
                first_token = self._sample_from_probs(first_probs, generator)

            generated: List[int] = [first_token]

            if first_token in self.eos_ids or max_new_tokens <= 1:
                self.last_target_cache = target_cache
                self.last_draft_cache = draft_cache
                output_ids = torch.cat(
                    [input_ids, torch.tensor([[first_token]], device=self.device)], dim=1
                )
                result = self._make_result(
                    track_metrics, ttft_timer, total_timer,
                    prompt_len, generated, 0, 0, 0, gamma
                )
                return output_ids, result

            current_token = torch.tensor([[first_token]], device=self.device)

            # Speculative decoding loop
            while len(generated) < max_new_tokens:
                remaining = max_new_tokens - len(generated)
                g = min(gamma, remaining)
                total_steps += 1
                total_proposed_draft += g

                # Cache length before step = prompt_len + len(generated) - 1
                prefix_cache_len = get_cache_seq_len(target_cache)

                # 1. Draft phase
                draft_tokens, draft_probs = self._draft_tokens(
                    current_token, draft_cache, g, temperature, top_p, generator
                )

                # 2. Verify phase (target parallel forward pass)
                target_probs = self._verify_tokens(
                    current_token, draft_tokens, target_cache, temperature, top_p
                )

                # 3. Rejection sampling
                accepted_tokens: List[int] = []
                correction_token: Optional[int] = None
                bonus_token: Optional[int] = None
                eos_hit = False

                for i in range(g):
                    d_i = draft_tokens[i]
                    p_t = target_probs[i]
                    p_d = draft_probs[i]

                    if temperature == 0.0:
                        t_tok = int(p_t.argmax())
                        if t_tok == d_i:
                            accepted_tokens.append(d_i)
                            if d_i in self.eos_ids:
                                eos_hit = True
                                break
                        else:
                            correction_token = t_tok
                            if correction_token in self.eos_ids:
                                eos_hit = True
                            break
                    else:
                        p_t_di = float(p_t[d_i].clamp(min=0.0))
                        p_d_di = float(p_d[d_i].clamp(min=1e-9))
                        accept_prob = min(1.0, p_t_di / p_d_di)

                        u = float(torch.rand(1, generator=generator).item())
                        if u <= accept_prob:
                            accepted_tokens.append(d_i)
                            if d_i in self.eos_ids:
                                eos_hit = True
                                break
                        else:
                            residual = self._adjusted_residual(p_t, p_d)
                            correction_token = self._sample_from_probs(residual, generator)
                            if correction_token in self.eos_ids:
                                eos_hit = True
                            break

                k = len(accepted_tokens)
                total_accepted_draft += k

                # Bonus token if all g accepted and no EOS
                if k == g and not eos_hit:
                    p_bonus = target_probs[g]
                    if temperature == 0.0:
                        bonus_token = int(p_bonus.argmax())
                    else:
                        bonus_token = self._sample_from_probs(p_bonus, generator)
                    if bonus_token in self.eos_ids:
                        eos_hit = True

                # Determine tokens emitted this step
                step_tokens = list(accepted_tokens)
                if correction_token is not None:
                    step_tokens.append(correction_token)
                elif bonus_token is not None:
                    step_tokens.append(bonus_token)

                # Append to generated sequence (respecting max_new_tokens)
                for tok in step_tokens:
                    generated.append(tok)
                    if len(generated) >= max_new_tokens:
                        break

                # 4. Cache rollback & synchronization
                final_cache_len = prompt_len + len(generated) - 1

                if k == g:
                    # All g draft tokens accepted:
                    # Advance draft cache by feeding the g-th draft token (draft_tokens[-1])
                    last_draft_tok = torch.tensor([[draft_tokens[-1]]], device=self.device)
                    draft_cache_pos = make_cache_position(get_cache_seq_len(draft_cache), 1, self.device)
                    _ = self.draft_model(
                        input_ids=last_draft_tok,
                        past_key_values=draft_cache,
                        use_cache=True,
                        cache_position=draft_cache_pos,
                    )

                # Crop both caches to exact final_cache_len (zero off-by-one or uncommitted token leaks)
                crop_cache(target_cache, final_cache_len)
                crop_cache(draft_cache, final_cache_len)

                if eos_hit or len(generated) >= max_new_tokens:
                    break

                current_token = torch.tensor([[generated[-1]]], device=self.device)

        self.last_target_cache = target_cache
        self.last_draft_cache = draft_cache

        output_ids = torch.cat(
            [input_ids, torch.tensor([generated], device=self.device)], dim=1
        )
        result = self._make_result(
            track_metrics, ttft_timer, total_timer,
            prompt_len, generated, total_accepted_draft, total_proposed_draft, total_steps, gamma
        )
        return output_ids, result

    def _make_result(
        self,
        track_metrics: bool,
        ttft_timer: LatencyTimer,
        total_timer: LatencyTimer,
        prompt_len: int,
        generated: List[int],
        total_accepted: int,
        total_proposed: int,
        total_steps: int,
        gamma: int,
    ) -> Optional[BenchmarkResult]:
        if not track_metrics:
            return None
        n_gen = len(generated)
        elapsed = total_timer.elapsed_s
        alpha = total_accepted / total_proposed if total_proposed > 0 else 0.0
        mean_per_step = n_gen / total_steps if total_steps > 0 else 0.0
        return BenchmarkResult(
            throughput_tps=n_gen / elapsed if elapsed > 0 else 0.0,
            speedup_ratio=1.0,
            ttft_ms=ttft_timer.elapsed_ms,
            elapsed_s=elapsed,
            total_tokens=n_gen,
            prompt_tokens=prompt_len,
            mean_acceptance_rate=alpha,
            mean_accepted_per_step=mean_per_step,
            total_steps=total_steps,
            gamma=gamma,
            peak_vram_mb=get_peak_vram_mb(self.device),
            decoder_type="speculative",
        )
