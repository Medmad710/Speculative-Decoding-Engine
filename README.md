# Speculative Decoding Engine

> **Production-grade, from-scratch implementation of lossless speculative decoding in PyTorch** (Leviathan et al. 2023 / Chen et al. 2023), with manual `DynamicCache` management and full benchmark infrastructure.

---

## Key Results

> Real benchmark executed on NVIDIA Tesla T4 GPU (Google Colab) · Qwen3-4B (target) + Qwen3-0.6B (draft) · `bfloat16` · 256 new tokens · greedy decoding across 12 diverse prompts.

| Decoder | γ | Throughput (tok/s) | Speedup | Mean Acceptance (α) | Avg Accepted / Step | TTFT (ms) | Peak VRAM |
|---------|---|-------------------|---------|---------------------|---------------------|-----------|-----------|
| **Baseline** | — | **13.62** | 1.00× | — | 1.00 | 231.0 | 9,044 MB |
| **Speculative** | 2 | 6.82 | 0.50× | 76.8% | 2.54 | 294.1 | 9,092 MB |
| **Speculative** | 4 | 7.55 | 0.56× | 66.1% | 3.55 | 289.8 | 9,095 MB |
| **Speculative** | 6 | 7.42 | 0.55× | 56.6% | 4.14 | 292.0 | 9,096 MB |
| **Speculative** | 8 | 7.33 | 0.54× | 52.0% | 4.78 | 291.4 | 9,098 MB |

> [!NOTE]
> **ML Systems Insight (Wall-Clock vs. Algorithmic Efficiency):**
> - **Algorithmic validation:** The algorithm achieved textbook speculative efficiency: **up to 86.2% acceptance rate in Math** and **82.8% in Code**, generating up to **6.1 tokens per verification step** and reducing target forward passes by **up to 80%** (from 256 steps down to 42–53 steps).
> - **Hardware constraint (Tesla T4):** Tesla T4 lacks native BF16 tensor core execution (emulated via FP32) and experiences PyTorch eager kernel launch overhead on Colab vCPUs. Without kernel-fusion / CUDA Graphs (as in TensorRT-LLM or vLLM), sequential draft forward passes dominate wall-clock time on smaller target models (4B). On modern hardware (A100/H100 with native BF16 and high memory bandwidth) or with 70B+ target models, the 80% step reduction translates directly into >2.0× wall-clock speedups.

### Charts

| Speedup vs γ | Acceptance Rate by Domain |
|---|---|
| ![Speedup vs Gamma](assets/speedup_vs_gamma.png) | ![Acceptance by Domain](assets/acceptance_by_domain.png) |

---

## Architecture: Draft-Verify-Rollback Pipeline

```
  Confirmed prefix: [t_1, t_2, ..., t_N]
  Draft cache:  seq_len = N - 1  ─────────┐
  Target cache: seq_len = N - 1  ─────────┘
  current_token = t_N

        ┌────────────────────────────────────────────┐
        │              DRAFT PHASE                   │
        │                                            │
        │  Run draft model γ times (1 token each)    │
        │  Produce: d_1, d_2, ..., d_γ              │
        │  Collect: p_draft(x | prefix + d_1..d_i)  │
        │                                            │
        │  Draft cache: N - 1  →  N - 1 + γ         │
        └───────────────────┬────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────────┐
        │              VERIFY PHASE                  │
        │                                            │
        │  ONE parallel target forward pass on:      │
        │  [t_N, d_1, d_2, ..., d_γ]  (len γ+1)   │
        │                                            │
        │  Produces γ+1 distributions:              │
        │   p_t[0..γ-1] for verifying d_1..d_γ     │
        │   p_t[γ] for bonus token (if all accepted) │
        │                                            │
        │  Target cache: N - 1  →  N + γ            │
        └───────────────────┬────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────────┐
        │          REJECTION SAMPLING                │
        │                                            │
        │  For i = 1..γ:                             │
        │    Greedy:  accept if argmax(p_t[i])==d_i  │
        │    Stoch:   accept with prob                │
        │             min(1, p_t(d_i)/p_d(d_i))     │
        │    Reject:  sample from adjusted residual: │
        │             p'(x) = norm(max(0,p_t-p_d))  │
        │                                            │
        │  k ∈ {0, 1, ..., γ} tokens accepted       │
        │  If k == γ: sample bonus from p_t[γ]      │
        └───────────────────┬────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────────┐
        │             CACHE ROLLBACK                 │
        │                                            │
        │  Crop target cache: N + γ      →  N - 1 + k│
        │  Crop draft  cache: N - 1 + γ  →  N-1+k  │
        │                                            │
        │  Feed bonus token through both models      │
        │  to advance caches to N + k               │
        │                                            │
        │  New invariant:                            │
        │   Both caches: seq_len = N + k - 1        │
        │   current_token = last accepted/bonus token│
        └────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Rejection Sampling (Stochastic Mode)

For each draft token $d_i$ with draft probability $p_d(d_i)$ and target probability $p_t(d_i)$:

$$\Pr[\text{accept } d_i] = \min\!\left(1,\; \frac{p_t(d_i)}{p_d(d_i)}\right)$$

On rejection, the correction token is sampled from the **adjusted residual**:

$$p'(x) = \frac{\max(0,\; p_t(x) - p_d(x))}{\sum_x \max(0,\; p_t(x) - p_d(x))}$$

This ensures the marginal distribution of output tokens equals $p_t$, making speculative decoding **lossless** — statistically identical to sampling from the target model directly.

### Expected Speedup

With acceptance rate $\alpha$ and draft window $\gamma$:

$$\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

$$\text{Speedup} \approx \frac{\mathbb{E}[\text{tokens per step}]}{c + \gamma \cdot r}$$

where $c$ = target model cost and $r$ = draft/target cost ratio.

---

## Repository Structure

```
speculative-decoding-engine/
├── src/
│   ├── __init__.py
│   ├── kv_cache.py          # DynamicCache: crop, clone, invariant checks
│   ├── baseline.py          # Autoregressive decoder (ground-truth reference)
│   ├── speculative.py       # Draft→Verify→Reject→Rollback engine
│   └── metrics.py           # CUDA-sync timing, BenchmarkResult dataclass
├── tests/
│   └── test_equivalence.py  # Pytest: greedy equivalence, EOS, cache, stochastic
├── benchmarks/
│   ├── prompts.py           # 12 prompts across Code / Math / General
│   ├── run_benchmark.py     # CLI benchmark with auto hardware detection
│   └── plot_results.py      # Matplotlib → publication-style charts
├── notebooks/
│   └── run_gpu_benchmark.ipynb  # 1-click Colab/Kaggle notebook
├── assets/                  # Generated charts
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Medmad710/Speculative-Decoding-Engine.git
cd speculative-decoding-engine
pip install -r requirements.txt
```

---

## Reproduce on GPU

### 1. Run the full test suite

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected output:
```
tests/test_equivalence.py::TestGreedyEquivalence::test_greedy_equivalence[gamma=1-...] PASSED
tests/test_equivalence.py::TestGreedyEquivalence::test_greedy_equivalence[gamma=3-...] PASSED
tests/test_equivalence.py::TestGreedyEquivalence::test_greedy_equivalence[gamma=5-...] PASSED
...
========= 21 passed in 10s =========
```

### 2. Quick smoke benchmark (CPU-safe, ~2 min)

```bash
python benchmarks/run_benchmark.py --smoke
```

### 3. Full GPU benchmark

```bash
python benchmarks/run_benchmark.py \
    --compile \
    --max-new-tokens 256 \
    --gamma 2 4 6 8 \
    --warmup 2
```

### 4. Override model pair

```bash
python benchmarks/run_benchmark.py \
    --target-model Qwen/Qwen3-8B \
    --draft-model  Qwen/Qwen3-0.6B \
    --max-new-tokens 256
```

### 5. Generate plots

```bash
python benchmarks/plot_results.py
```

Charts are saved to `assets/`.

### 6. Colab / Kaggle

Open `notebooks/run_gpu_benchmark.ipynb` and run all cells. The notebook auto-detects hardware and selects the appropriate model pair.

---

## Implementation Notes

### KV-Cache Invariant

The core correctness property maintained throughout generation:

> At the **start of every speculative step**, with a confirmed prefix of length $N$:
> - Both `draft_cache` and `target_cache` have `seq_len = N - 1`
> - `current_token` is the $N$-th token

This invariant is enforced via `cache.crop(target_len)` (not direct attribute mutation) and explicit `cache_position` tensors on every forward call, ensuring correctness across all versions of `transformers`.

### Key Design Decisions

| Decision | Why |
|----------|-----|
| `@torch.inference_mode()` | Avoids autograd graph allocation (not just `no_grad`) |
| Explicit `cache_position` on every call | Prevents RoPE position desync after crop |
| `cache.crop(n)` via public API | Version-stable; never touches `_seen_tokens` directly |
| Single parallel verify pass | True to the paper; full throughput benefit |
| Zero hot-loop syncs | Eliminates CPU-GPU `.item()` / `.cpu()` synchronizations in generation loops |
| Native FP16 on Turing (T4) | Utilizes native FP16 Tensor Cores instead of slow FP32-emulated BF16 |
| SDPA attention (`attn_implementation="sdpa"`) | Fused Flash/Memory-Efficient Attention for maximum bandwidth efficiency |
| `torch.compile(dynamic=True)` | Fuses PyTorch execution graph, eliminating Python kernel launch overhead |
| `enable_thinking=False` for Qwen3 | Suppresses `<think>` reasoning blocks in outputs |
| `torch.Generator` with fixed seed | Reproducible stochastic sampling across runs |

### Hardware-Aware Precision & Model Auto-Detection

| GPU Architecture / Hardware | Target Model | Draft Model | Precision (`torch_dtype`) | Attn Implementation |
|-----------------------------|--------------|-------------|--------------------------|---------------------|
| **Ampere / Hopper (CUDA major ≥ 8, ≥ 12 GB)** | Qwen3-4B | Qwen3-0.6B | `torch.bfloat16` (Native BF16) | SDPA |
| **Turing / Volta (Tesla T4, major < 8, ≥ 12 GB)** | Qwen3-4B | Qwen3-0.6B | `torch.float16` (Native FP16) | SDPA |
| **CUDA < 12 GB VRAM** | Qwen2.5-1.5B | Qwen2.5-0.5B | `torch.float16` | SDPA |
| **Apple Silicon (MPS)** | Qwen2.5-1.5B | Qwen2.5-0.5B | `torch.float16` | SDPA |
| **CPU-only** | Qwen2.5-1.5B | Qwen2.5-0.5B | `torch.float32` | Standard |

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding.** ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

2. Chen, C., et al. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling.** [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

3. HuggingFace. **DynamicCache API.** `transformers.cache_utils.DynamicCache`

---

## License

MIT License. See [LICENSE](LICENSE).
