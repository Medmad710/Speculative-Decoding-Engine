"""
metrics.py — Benchmark Metrics & Timing Utilities
==================================================
Provides CUDA-synchronized timing, throughput calculation, and a structured
result dataclass for comparing Baseline vs. Speculative decoding.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import torch


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Structured benchmark result for one decoder run."""

    # Core timing
    throughput_tps: float = 0.0        # tokens generated per second
    speedup_ratio: float = 1.0         # relative to baseline (baseline = 1.0)
    ttft_ms: float = 0.0               # time-to-first-token in milliseconds
    elapsed_s: float = 0.0             # total wall-clock time (seconds)

    # Token accounting
    total_tokens: int = 0              # number of new tokens generated
    prompt_tokens: int = 0             # number of input (prompt) tokens

    # Speculative decoding statistics (None for baseline)
    mean_acceptance_rate: Optional[float] = None   # mean α per step
    mean_accepted_per_step: Optional[float] = None # mean tokens accepted per step
    total_steps: Optional[int] = None              # total speculative steps
    gamma: Optional[int] = None                    # draft window size used

    # Hardware
    peak_vram_mb: float = 0.0          # peak GPU memory allocated (MB)

    # Domain / metadata
    domain: str = ""
    prompt_preview: str = ""           # first 60 chars of prompt
    decoder_type: str = "baseline"     # "baseline" or "speculative"

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict."""
        return {
            "decoder_type": self.decoder_type,
            "domain": self.domain,
            "prompt_preview": self.prompt_preview,
            "gamma": self.gamma,
            "throughput_tps": round(self.throughput_tps, 3),
            "speedup_ratio": round(self.speedup_ratio, 3),
            "ttft_ms": round(self.ttft_ms, 3),
            "elapsed_s": round(self.elapsed_s, 4),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "mean_acceptance_rate": (
                round(self.mean_acceptance_rate, 4)
                if self.mean_acceptance_rate is not None else None
            ),
            "mean_accepted_per_step": (
                round(self.mean_accepted_per_step, 4)
                if self.mean_accepted_per_step is not None else None
            ),
            "total_steps": self.total_steps,
            "peak_vram_mb": round(self.peak_vram_mb, 2),
        }


# ---------------------------------------------------------------------------
# CUDA-synchronized timer
# ---------------------------------------------------------------------------

class LatencyTimer:
    """Context manager that measures wall-clock time with CUDA synchronization.

    On CUDA devices, ``torch.cuda.synchronize()`` is called before starting
    and after stopping to ensure all GPU kernels have finished.

    Usage::

        timer = LatencyTimer(device)
        with timer:
            output = model(...)
        print(f"Elapsed: {timer.elapsed_ms:.2f} ms")
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed_ms: float = 0.0
        self.elapsed_s: float = 0.0
        self._start: float = 0.0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self) -> "LatencyTimer":
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self._sync()
        self.elapsed_s = time.perf_counter() - self._start
        self.elapsed_ms = self.elapsed_s * 1000.0


@contextmanager
def cuda_timer(device: torch.device) -> Generator[LatencyTimer, None, None]:
    """Convenience context manager returning a ``LatencyTimer``."""
    t = LatencyTimer(device)
    t._sync()
    t._start = time.perf_counter()
    yield t
    t._sync()
    t.elapsed_s = time.perf_counter() - t._start
    t.elapsed_ms = t.elapsed_s * 1000.0


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def reset_peak_vram(device: torch.device) -> None:
    """Reset peak memory stats on CUDA; no-op elsewhere."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_vram_mb(device: torch.device) -> float:
    """Return peak GPU memory allocated in MB; 0.0 on CPU/MPS."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_speedup(baseline: BenchmarkResult, speculative: BenchmarkResult) -> float:
    """Return tokens/sec speedup ratio (speculative / baseline)."""
    if baseline.throughput_tps == 0:
        return 0.0
    return speculative.throughput_tps / baseline.throughput_tps


def aggregate_results(results: List[BenchmarkResult]) -> dict:
    """Compute mean metrics over a list of results (same decoder type)."""
    if not results:
        return {}
    tps = [r.throughput_tps for r in results]
    speedup = [r.speedup_ratio for r in results]
    alphas = [r.mean_acceptance_rate for r in results if r.mean_acceptance_rate is not None]
    return {
        "mean_throughput_tps": sum(tps) / len(tps),
        "mean_speedup": sum(speedup) / len(speedup),
        "mean_acceptance_rate": sum(alphas) / len(alphas) if alphas else None,
        "n": len(results),
    }
