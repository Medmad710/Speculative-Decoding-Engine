"""
conftest.py — Shared pytest configuration
"""
import os
import sys
from pathlib import Path

# Always add project root to sys.path so src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
