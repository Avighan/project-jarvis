"""
Project Jarvis PoC — Ollama local API wrapper.
All calls go to localhost:11434. Zero cloud.
"""

import json
import time
import requests
from typing import Optional

OLLAMA_BASE  = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"
EMBED_MODEL   = "nomic-embed-text"


def available_models() -> list[str]:
    """Return list of pulled model names."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pick_model(preferred: str = DEFAULT_MODEL) -> str:
    """Return preferred model if available, else first available model."""
    models = available_models()
    if not models:
        raise RuntimeError("Ollama is not running or no models are pulled. "
                           "Start Ollama and run: ollama pull llama3")
    if preferred in models:
        return preferred
    print(f"[warn] {preferred} not found. Using {models[0]} instead.")
    return models[0]


def generate(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> tuple[str, int]:
    """
    Generate a response from the local Ollama model.
    Returns (response_text, latency_ms).
    """
    model = model or pick_model()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    if system:
        payload["system"] = system

    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        latency_ms = int((time.time() - start) * 1000)
        return data.get("response", ""), latency_ms
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out. Is the model loaded?")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama at localhost:11434. "
                           "Start Ollama with: ollama serve")


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """
    Generate a vector embedding for text using local Ollama embedding model.
    Requires: ollama pull nomic-embed-text
    Returns list of floats (embedding vector).
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except requests.exceptions.HTTPError as e:
        if "model" in str(e).lower() or "404" in str(e):
            raise RuntimeError(
                f"Embedding model '{model}' not found. "
                f"Pull it with: ollama pull {model}"
            )
        raise


def classify(
    query: str,
    categories: list[str],
    model: Optional[str] = None,
) -> str:
    """
    Classify query into one of the given categories using local model.
    Returns the winning category name (stripped, uppercased).
    """
    model = model or pick_model()
    cats = " | ".join(categories)
    prompt = (
        f"Classify this request into exactly one of these categories:\n"
        f"{cats}\n\n"
        f"Request: {query}\n\n"
        f"Reply with only the category name. Nothing else."
    )
    response, _ = generate(prompt, model=model, temperature=0.0, max_tokens=20)
    result = response.strip().upper()
    # Find closest match in categories
    for cat in categories:
        if cat.upper() in result:
            return cat.upper()
    return result  # return raw if no match found


if __name__ == "__main__":
    print("Available models:", available_models())
    model = pick_model()
    print(f"Using: {model}")
    response, ms = generate("Say hello in one sentence.", model=model)
    print(f"Response ({ms}ms): {response}")
