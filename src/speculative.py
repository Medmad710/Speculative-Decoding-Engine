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
        compile:        If True, compile models with torch.compile(dynamic=True).
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
        compile: bool = False,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device or next(target_model.parameters()).device
        self.compiled = False

        if compile:
            try:
                self.draft_model = torch.compile(self.draft_model, dynamic=True)
                self.target_model = torch.compile(self.target_model, dynamic=True)
                self.compiled = True
            except Exception as e:
                self.compiled = False

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
    ) -> torch.Tensor:
        return torch.multinomial(probs, num_samples=1, generator=generator)

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
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate gamma draft tokens asynchronously on device without per-token CPU round trips."""
        draft_token_tensors: List[torch.Tensor] = []
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
            if temperature == 0.0:
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                logits = out.logits[0, -1, :]
                probs = self._probs_from_logits(logits, temperature, top_p)
                next_tok = self._sample_from_probs(probs, generator).view(1, 1)
                draft_probs.append(probs)

            draft_token_tensors.append(next_tok)
            inp = next_tok

        draft_tokens_tensor = torch.cat(draft_token_tensors, dim=1)  # (1, gamma)
        return draft_tokens_tensor, draft_probs

    @torch.inference_mode()
    def _verify_tokens(
        self,
        current_token: torch.Tensor,
        draft_tokens_tensor: torch.Tensor,
        target_cache: DynamicCache,
    ) -> torch.Tensor:
        """Run single parallel verification forward pass on device."""
        all_tokens = torch.cat([current_token, draft_tokens_tensor], dim=1)  # (1, gamma + 1)
        seq_in_cache = get_cache_seq_len(target_cache)
        gamma_plus_1 = all_tokens.shape[1]
        cache_pos = make_cache_position(seq_in_cache, gamma_plus_1, self.device)

        out = self.target_model(
            input_ids=all_tokens,
            past_key_values=target_cache,
            use_cache=True,
            cache_position=cache_pos,
        )
        return out.logits[0]  # (gamma + 1, vocab)

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
        """Run speculative decoding with zero per-token CPU round trips."""
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

        eos_tensor = torch.tensor(list(self.eos_ids), device=self.device, dtype=torch.long)

        with total_timer:
            # Prefill phase
            with ttft_timer:
                draft_cache, target_cache, prefill_logits = self._prefill(input_ids)

            if temperature == 0.0:
                first_token = prefill_logits.argmax().view(1, 1)
            else:
                first_probs = self._probs_from_logits(prefill_logits, temperature, top_p)
                first_token = self._sample_from_probs(first_probs, generator).view(1, 1)

            generated_tensors: List[torch.Tensor] = [first_token.view(-1)]
            current_generated_len = 1

            if bool(torch.isin(first_token, eos_tensor).item()) or max_new_tokens <= 1:
                self.last_target_cache = target_cache
                self.last_draft_cache = draft_cache
                all_generated = torch.cat(generated_tensors, dim=0)
                output_ids = torch.cat([input_ids, all_generated.unsqueeze(0)], dim=1)
                generated_list = all_generated.cpu().tolist()
                result = self._make_result(
                    track_metrics, ttft_timer, total_timer,
                    prompt_len, generated_list, 0, 0, 0, gamma
                )
                return output_ids, result

            current_token = first_token  # shape (1, 1) on device

            # Speculative decoding loop (hot loop kept on device)
            while current_generated_len < max_new_tokens:
                remaining = max_new_tokens - current_generated_len
                g = min(gamma, remaining)
                total_steps += 1
                total_proposed_draft += g

                # 1. Draft phase
                draft_tokens_tensor, draft_probs = self._draft_tokens(
                    current_token, draft_cache, g, temperature, top_p, generator
                )

                # 2. Verify phase (target parallel forward pass)
                target_logits = self._verify_tokens(
                    current_token, draft_tokens_tensor, target_cache
                )

                # 3. Rejection sampling / verification
                if temperature == 0.0:
                    target_preds = target_logits.argmax(dim=-1)  # (g + 1,)
                    draft_1d = draft_tokens_tensor[0]            # (g,)
                    matches = (draft_1d == target_preds[:g])     # (g,) bool

                    cum_matches = torch.cumprod(matches.to(torch.long), dim=0)
                    k = int(cum_matches.sum().item())
                    total_accepted_draft += k

                    # Token at position k in target_preds is the next token:
                    # bonus token if k == g, or correction token if k < g
                    step_tokens = torch.cat([draft_1d[:k], target_preds[k : k + 1]])

                else:
                    target_probs: List[torch.Tensor] = []
                    for i in range(g + 1):
                        target_probs.append(self._probs_from_logits(target_logits[i], temperature, top_p))

                    accepted_tokens: List[torch.Tensor] = []
                    correction_token: Optional[torch.Tensor] = None
                    bonus_token: Optional[torch.Tensor] = None
                    eos_hit_stoch = False

                    for i in range(g):
                        d_i = draft_tokens_tensor[0, i : i + 1]
                        d_i_int = int(d_i.item())
                        p_t = target_probs[i]
                        p_d = draft_probs[i]

                        p_t_di = float(p_t[d_i_int].clamp(min=0.0))
                        p_d_di = float(p_d[d_i_int].clamp(min=1e-9))
                        accept_prob = min(1.0, p_t_di / p_d_di)

                        u = float(torch.rand(1, generator=generator).item())
                        if u <= accept_prob:
                            accepted_tokens.append(d_i)
                            if d_i_int in self.eos_ids:
                                eos_hit_stoch = True
                                break
                        else:
                            residual = self._adjusted_residual(p_t, p_d)
                            corr = self._sample_from_probs(residual, generator)
                            correction_token = corr.view(1)
                            if int(correction_token.item()) in self.eos_ids:
                                eos_hit_stoch = True
                            break

                    k = len(accepted_tokens)
                    total_accepted_draft += k

                    if k == g and not eos_hit_stoch:
                        p_bonus = target_probs[g]
                        bonus_token = self._sample_from_probs(p_bonus, generator).view(1)

                    step_parts = list(accepted_tokens)
                    if correction_token is not None:
                        step_parts.append(correction_token)
                    elif bonus_token is not None:
                        step_parts.append(bonus_token)
                    step_tokens = torch.cat(step_parts, dim=0)

                # 4. Check EOS on step_tokens
                is_eos = torch.isin(step_tokens, eos_tensor)
                if bool(is_eos.any().item()):
                    eos_idx = int(torch.nonzero(is_eos)[0, 0].item())
                    step_tokens = step_tokens[: eos_idx + 1]
                    eos_hit = True
                else:
                    eos_hit = False

                # 5. Respect max_new_tokens
                if current_generated_len + step_tokens.shape[0] > max_new_tokens:
                    allowed = max_new_tokens - current_generated_len
                    step_tokens = step_tokens[:allowed]

                num_emitted = step_tokens.shape[0]
                current_generated_len += num_emitted
                final_cache_len = prompt_len + current_generated_len - 1

                # 6. Cache rollback & synchronization
                if k == g:
                    last_draft_tok = draft_tokens_tensor[0, -1:].view(1, 1)
                    draft_pos = make_cache_position(get_cache_seq_len(draft_cache), 1, self.device)
                    _ = self.draft_model(
                        input_ids=last_draft_tok,
                        past_key_values=draft_cache,
                        use_cache=True,
                        cache_position=draft_pos,
                    )

                crop_cache(target_cache, final_cache_len)
                crop_cache(draft_cache, final_cache_len)

                generated_tensors.append(step_tokens)
                current_token = step_tokens[-1:].view(1, 1)

                if eos_hit or current_generated_len >= max_new_tokens:
                    break

        self.last_target_cache = target_cache
        self.last_draft_cache = draft_cache

        all_generated = torch.cat(generated_tensors, dim=0)
        output_ids = torch.cat([input_ids, all_generated.unsqueeze(0)], dim=1)
        generated_list = all_generated.cpu().tolist()

        result = self._make_result(
            track_metrics, ttft_timer, total_timer,
            prompt_len, generated_list, total_accepted_draft, total_proposed_draft, total_steps, gamma
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
