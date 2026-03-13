"""Shared Ollama helpers for all experiments."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from core.ollama_client import generate, embed, classify, pick_model, available_models
__all__ = ["generate", "embed", "classify", "pick_model", "available_models"]
