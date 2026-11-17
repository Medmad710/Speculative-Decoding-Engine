"""
Speculative Decoding Engine
===========================
Production-grade implementation of lossless speculative decoding
(Leviathan et al. 2023 / Chen et al. 2023) in PyTorch with manual
DynamicCache management.
"""
from .baseline import BaselineDecoder
from .speculative import SpeculativeDecoder
from .metrics import BenchmarkResult, LatencyTimer

__all__ = ["BaselineDecoder", "SpeculativeDecoder", "BenchmarkResult", "LatencyTimer"]
