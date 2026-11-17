"""
prompts.py — Curated Evaluation Prompt Bank
============================================
Three domains × 4 prompts = 12 benchmark prompts, covering diverse
generation patterns that stress-test speculative decoding acceptance rates.

Domain characteristics:
  Code       — High acceptance rate (repetitive syntax, predictable structure)
  Math       — Medium acceptance rate (formulaic but varied symbol sequences)
  General    — Lower acceptance rate (more diverse vocabulary)
"""

from __future__ import annotations
from typing import List, Tuple

# Each entry: (domain, prompt_text)
BENCHMARK_PROMPTS: List[Tuple[str, str]] = [
    # ---- Code: Python ----
    (
        "code",
        "Write a Python function that implements merge sort on a list of integers. "
        "Include type hints and a docstring.",
    ),
    (
        "code",
        "Write a Python class implementing a doubly linked list with insert, delete, "
        "and reverse methods. Include full docstrings.",
    ),
    # ---- Code: C++ ----
    (
        "code",
        "Implement a binary search tree in C++ with insert, search, and in-order "
        "traversal methods. Use templates.",
    ),
    (
        "code",
        "Write a C++ function that uses dynamic programming to solve the 0/1 knapsack "
        "problem. Include comments.",
    ),
    # ---- Math / Reasoning ----
    (
        "math",
        "Derive the formula for the sum of a geometric series from first principles. "
        "Show all steps clearly.",
    ),
    (
        "math",
        "Prove that the square root of 2 is irrational using proof by contradiction. "
        "Be rigorous.",
    ),
    (
        "math",
        "Solve the following integral step by step: integral of x * e^x dx from 0 to 1. "
        "Use integration by parts.",
    ),
    (
        "math",
        "Explain the chain rule in calculus with three worked examples of increasing "
        "complexity.",
    ),
    # ---- General Text ----
    (
        "general",
        "Write a detailed explanation of how transformer attention mechanisms work, "
        "suitable for a machine learning engineer.",
    ),
    (
        "general",
        "Summarize the key differences between supervised, unsupervised, and "
        "reinforcement learning in machine learning.",
    ),
    (
        "general",
        "Explain the CAP theorem in distributed systems and give a real-world example "
        "of each trade-off.",
    ),
    (
        "general",
        "Describe the main architectural differences between GPT-style and BERT-style "
        "language models.",
    ),
]


def get_prompts_by_domain(domain: str) -> List[str]:
    """Return all prompts for a given domain."""
    return [p for d, p in BENCHMARK_PROMPTS if d == domain]


def get_all_prompts() -> List[Tuple[str, str]]:
    """Return all (domain, prompt) pairs."""
    return BENCHMARK_PROMPTS
