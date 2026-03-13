"""
Project Jarvis PoC — Retrieval methods.
Method A: TF-IDF keyword scoring (numpy only — zero extra deps).
Method B: Cosine similarity on Ollama embeddings (requires nomic-embed-text).
"""

import math
import re
import numpy as np
from typing import Optional


# ── Shared text helpers ───────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","i","me",
    "my","we","our","you","your","he","she","it","they","their","this","that",
    "these","those","what","how","why","when","where","who","which"
}


def tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_idf(corpus: list[str]) -> dict[str, float]:
    """Compute IDF for each term across the corpus."""
    N = len(corpus)
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(tokenise(doc)):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def tfidf_score(query: str, document: str, idf: dict[str, float]) -> float:
    """
    Score relevance of document to query using TF-IDF dot product.
    Returns a non-negative float; higher = more relevant.
    """
    query_terms = tokenise(query)
    doc_tokens  = tokenise(document)
    if not query_terms or not doc_tokens:
        return 0.0

    doc_tf: dict[str, float] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    doc_len = len(doc_tokens)

    score = 0.0
    for term in query_terms:
        tf  = doc_tf.get(term, 0) / doc_len
        idf_val = idf.get(term, 1.0)
        score += tf * idf_val
    return score


# ── Method A: TF-IDF retrieval ────────────────────────────────────────────────

def retrieve_tfidf(
    query: str,
    memories: list[dict],
    top_n: int = 4,
    confidence_weight: bool = False,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Retrieve top-N memories by TF-IDF relevance.
    memories: list of dicts with at least 'content' and 'confidence' keys.
    confidence_weight: if True, multiply TF-IDF score by memory confidence.
    Returns list of dicts with added 'retrieval_score' key, sorted desc.
    """
    if not memories:
        return []

    corpus = [m["content"] for m in memories]
    idf    = build_idf(corpus + [query])

    scored = []
    for mem in memories:
        score = tfidf_score(query, mem["content"], idf)
        if confidence_weight:
            score *= mem.get("confidence", 1.0)
        if score > min_score:
            scored.append({**mem, "retrieval_score": score})

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_n]


# ── Method B: Cosine similarity on local embeddings ──────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def retrieve_embeddings(
    query_embedding: list[float],
    memories: list[dict],
    top_n: int = 4,
    confidence_weight: bool = False,
) -> list[dict]:
    """
    Retrieve top-N memories by cosine similarity against pre-computed embeddings.
    memories: list of dicts; each must have 'embedding' key (list of floats).
    Memories without embeddings are skipped.
    """
    scored = []
    for mem in memories:
        if not mem.get("embedding"):
            continue
        emb = mem["embedding"]
        if isinstance(emb, (bytes, bytearray)):
            import struct
            n = len(emb) // 4
            emb = list(struct.unpack(f"{n}f", emb))
        score = cosine_similarity(query_embedding, emb)
        if confidence_weight:
            score *= mem.get("confidence", 1.0)
        scored.append({**mem, "retrieval_score": score})

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_n]


# ── Disinhibition: task-type gating ──────────────────────────────────────────

TASK_SUPPRESSION = {
    "PLANNING":    {"suppress": ["preference"], "boost": ["goal", "pattern"]},
    "RESEARCH":    {"suppress": ["preference"], "boost": ["expertise", "general"]},
    "EXECUTION":   {"suppress": ["goal", "pattern"], "boost": ["general"]},
    "EMOTIONAL":   {"suppress": ["expertise", "general"], "boost": ["pattern"]},
    "REFLECTION":  {"suppress": ["general"], "boost": ["pattern", "goal"]},
    "LEARNING":    {"suppress": ["preference"], "boost": ["expertise"]},
}


def retrieve_disinhibition(
    query: str,
    task_type: str,
    memories: list[dict],
    top_n: int = 4,
    confidence_weight: bool = False,
) -> list[dict]:
    """
    Disinhibition retrieval: suppress irrelevant categories, boost relevant ones,
    then apply TF-IDF within the filtered set.
    """
    rules = TASK_SUPPRESSION.get(task_type.upper(), {})
    suppress = set(rules.get("suppress", []))
    boost    = set(rules.get("boost", []))

    # Filter out suppressed categories
    filtered = [m for m in memories if m.get("category", "general") not in suppress]

    if not filtered:
        filtered = memories  # fallback: don't return empty

    corpus = [m["content"] for m in filtered]
    idf    = build_idf(corpus + [query])

    scored = []
    for mem in filtered:
        score = tfidf_score(query, mem["content"], idf)
        # Boost score for preferred categories
        if mem.get("category", "general") in boost:
            score *= 1.5
        if confidence_weight:
            score *= mem.get("confidence", 1.0)
        scored.append({**mem, "retrieval_score": score})

    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_n]
