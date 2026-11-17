"""
run_benchmark.py — CLI Benchmark: Baseline vs. Speculative Decoding
====================================================================
Compares throughput, TTFT, acceptance rate and peak VRAM across
gamma values and prompt domains.

Usage
-----
  python benchmarks/run_benchmark.py [OPTIONS]

Options
-------
  --target-model  MODEL_ID     Override target model (default: auto-detect)
  --draft-model   MODEL_ID     Override draft model (default: auto-detect)
  --max-new-tokens INT         Tokens to generate per prompt (default: 256)
  --gamma         INT [INT...] Gamma values to sweep (default: 2 4 6 8)
  --warmup        INT          Warmup runs before timing (default: 2)
  --output        PATH         JSON output path (default: benchmarks/results.json)
  --smoke                      Quick smoke test: 1 prompt, gamma=[2,4], max=64
  --temperature   FLOAT        Sampling temperature (default: 0.0)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Allow running from project root or benchmarks/ directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.baseline import BaselineDecoder
from src.speculative import SpeculativeDecoder
from src.metrics import BenchmarkResult, compute_speedup, get_peak_vram_mb, reset_peak_vram
from benchmarks.prompts import get_all_prompts


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> Tuple[torch.device, str, str, torch.dtype]:
    """Detect available hardware and choose appropriate model pair.

    Returns:
        (device, target_model_id, draft_model_id, dtype)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[hw] CUDA detected — GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {total_vram_gb:.1f} GB")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if total_vram_gb >= 12.0:
            target = "Qwen/Qwen3-4B"
            draft = "Qwen/Qwen3-0.6B"
            print(f"[hw] ≥12 GB VRAM → using {target} + {draft}")
        else:
            target = "Qwen/Qwen2.5-1.5B-Instruct"
            draft = "Qwen/Qwen2.5-0.5B-Instruct"
            print(f"[hw] <12 GB VRAM → using {target} + {draft}")

    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        target = "Qwen/Qwen2.5-1.5B-Instruct"
        draft = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[hw] Apple MPS detected → using {target} + {draft}")

    else:
        device = torch.device("cpu")
        dtype = torch.float32
        target = "Qwen/Qwen2.5-1.5B-Instruct"
        draft = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[hw] CPU-only → using {target} + {draft}")

    return device, target, draft, dtype


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    target_model_id: str,
    draft_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load tokenizer and both models."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n[load] Tokenizer from {target_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] Draft model {draft_model_id}...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id, torch_dtype=dtype
    ).to(device).eval()

    print(f"[load] Target model {target_model_id}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, torch_dtype=dtype
    ).to(device).eval()

    print("[load] Models ready.\n")
    return tokenizer, draft_model, target_model


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Apply chat template and return input_ids tensor."""
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
    return ids.to(device)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def run_warmup(
    baseline: BaselineDecoder,
    speculative: SpeculativeDecoder,
    tokenizer,
    device: torch.device,
    n_warmup: int,
    max_new_tokens: int,
):
    """Run warmup generations to prime CUDA kernels and cache."""
    warmup_prompt = "Hello, how are you?"
    input_ids = apply_chat_template(tokenizer, warmup_prompt, device)

    print(f"[warmup] Running {n_warmup} warmup generation(s)...")
    for i in range(n_warmup):
        baseline.generate(input_ids.clone(), max_new_tokens=min(32, max_new_tokens))
        speculative.generate(input_ids.clone(), max_new_tokens=min(32, max_new_tokens), gamma=4)
        if device.type == "cuda":
            torch.cuda.synchronize()
    print("[warmup] Done.\n")


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def benchmark_one_prompt(
    baseline: BaselineDecoder,
    speculative: SpeculativeDecoder,
    tokenizer,
    domain: str,
    prompt: str,
    gamma_values: List[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> Tuple[BenchmarkResult, List[BenchmarkResult]]:
    """Run baseline + speculative for all gamma values on one prompt.

    Returns:
        (baseline_result, [spec_result_gamma2, spec_result_gamma4, ...])
    """
    input_ids = apply_chat_template(tokenizer, prompt, device)
    prompt_preview = prompt[:60]

    # ---- Baseline ----
    if device.type == "cuda":
        torch.cuda.synchronize()
        reset_peak_vram(device)

    baseline_result, _ = baseline.generate(
        input_ids.clone(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        track_metrics=True,
    )
    b_result = _
    b_result.domain = domain
    b_result.prompt_preview = prompt_preview

    baseline_tps = b_result.throughput_tps

    # ---- Speculative (each gamma) ----
    spec_results: List[BenchmarkResult] = []
    for gamma in gamma_values:
        if device.type == "cuda":
            torch.cuda.synchronize()
            reset_peak_vram(device)

        _, s_result = speculative.generate(
            input_ids.clone(),
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            temperature=temperature,
            track_metrics=True,
        )
        s_result.domain = domain
        s_result.prompt_preview = prompt_preview
        s_result.speedup_ratio = compute_speedup(b_result, s_result)
        s_result.gamma = gamma
        spec_results.append(s_result)

        print(
            f"  gamma={gamma:2d} | "
            f"tps={s_result.throughput_tps:7.1f} | "
            f"speedup={s_result.speedup_ratio:.2f}x | "
            f"α={s_result.mean_acceptance_rate:.3f} | "
            f"TTFT={s_result.ttft_ms:.1f}ms | "
            f"VRAM={s_result.peak_vram_mb:.0f}MB"
        )

    return b_result, spec_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=str, default=str(_ROOT / "benchmarks" / "results.json"))
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 1 prompt, gamma=[2,4], max=64")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Hardware
    device, auto_target, auto_draft, dtype = detect_hardware()
    target_model_id = args.target_model or auto_target
    draft_model_id = args.draft_model or auto_draft

    # Load models
    tokenizer, draft_model, target_model = load_models(
        target_model_id, draft_model_id, device, dtype
    )

    # Instantiate decoders
    baseline = BaselineDecoder(target_model, tokenizer, device=device)
    speculative = SpeculativeDecoder(draft_model, target_model, tokenizer, device=device)

    # Benchmark params
    gamma_values = [2, 4] if args.smoke else args.gamma
    max_new_tokens = 64 if args.smoke else args.max_new_tokens
    prompts = get_all_prompts()
    if args.smoke:
        prompts = [prompts[0]]  # just the first code prompt

    # Warmup
    run_warmup(baseline, speculative, tokenizer, device, args.warmup, max_new_tokens)

    # Run benchmark
    all_baseline: List[dict] = []
    all_spec: List[dict] = []

    total_prompts = len(prompts)
    for idx, (domain, prompt) in enumerate(prompts):
        print(f"\n[{idx+1}/{total_prompts}] Domain={domain} | "
              f"Prompt: '{prompt[:50]}...'")

        b_result, s_results = benchmark_one_prompt(
            baseline, speculative, tokenizer,
            domain, prompt, gamma_values,
            max_new_tokens, args.temperature, device,
        )

        print(
            f"  BASELINE | tps={b_result.throughput_tps:7.1f} | "
            f"TTFT={b_result.ttft_ms:.1f}ms | "
            f"VRAM={b_result.peak_vram_mb:.0f}MB"
        )

        all_baseline.append(b_result.to_dict())
        for s in s_results:
            all_spec.append(s.to_dict())

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "target_model": target_model_id,
            "draft_model": draft_model_id,
            "device": str(device),
            "dtype": str(dtype),
            "max_new_tokens": max_new_tokens,
            "gamma_values": gamma_values,
            "temperature": args.temperature,
        },
        "baseline": all_baseline,
        "speculative": all_spec,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY (mean across all prompts)")
    print("=" * 70)
    print(f"{'Gamma':>6} | {'TPS':>8} | {'Speedup':>8} | {'Alpha':>7} | {'TTFT(ms)':>10}")
    print("-" * 70)
    for gamma in gamma_values:
        g_results = [r for r in all_spec if r["gamma"] == gamma]
        if g_results:
            mean_tps = sum(r["throughput_tps"] for r in g_results) / len(g_results)
            mean_speedup = sum(r["speedup_ratio"] for r in g_results) / len(g_results)
            alphas = [r["mean_acceptance_rate"] for r in g_results if r["mean_acceptance_rate"]]
            mean_alpha = sum(alphas) / len(alphas) if alphas else 0.0
            mean_ttft = sum(r["ttft_ms"] for r in g_results) / len(g_results)
            print(f"  γ={gamma:<4d} | {mean_tps:>8.1f} | {mean_speedup:>7.2f}x | "
                  f"{mean_alpha:>7.3f} | {mean_ttft:>10.1f}")


if __name__ == "__main__":
    main()
