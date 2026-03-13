"""
Experiment 5 — Retrieval: TF-IDF Keyword vs. Vector Embeddings
===============================================================
50 memories, 20 queries with known correct answers.
Method A: TF-IDF (numpy only — zero deps)
Method B: Cosine similarity on nomic-embed-text (requires: ollama pull nomic-embed-text)

Measures: Recall@1, Recall@5 for each method.
Decision: if TF-IDF Recall@5 >= 75%, use it for PoC; add embeddings in Phase 2.

Run: python3 POC/experiments/experiment_05/run.py
Method B requires: ollama pull nomic-embed-text
"""

import sys, json, pathlib, datetime, struct
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model, available_models
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from core.retrieval             import retrieve_tfidf, cosine_similarity

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
EMBED_MODEL = "nomic-embed-text"

# ── 50 memories ──────────────────────────────────────────────────────────────
MEMORIES_50 = [
    # ID 1-10: Engineering preferences & style
    {"content": "user prefers direct answers without preamble", "category": "preference", "confidence": 0.9},
    {"content": "user writes Python with strict type hints", "category": "preference", "confidence": 0.8},
    {"content": "user prefers minimal dependencies in production code", "category": "preference", "confidence": 0.85},
    {"content": "user uses pytest for testing Python code", "category": "preference", "confidence": 0.75},
    {"content": "user prefers SQLite over Postgres for local projects", "category": "preference", "confidence": 0.8},
    {"content": "user avoids global state in Python modules", "category": "preference", "confidence": 0.7},
    {"content": "user writes docstrings only for public APIs", "category": "preference", "confidence": 0.65},
    {"content": "user prefers composition over inheritance in Python", "category": "preference", "confidence": 0.75},
    {"content": "user uses dataclasses for structured data in Python", "category": "preference", "confidence": 0.7},
    {"content": "user prefers synchronous code for PoC work", "category": "preference", "confidence": 0.8},
    # ID 11-20: Project and goals
    {"content": "user is building Project Jarvis — a local AI memory layer", "category": "goal", "confidence": 0.95},
    {"content": "user wants Jarvis to work completely offline", "category": "goal", "confidence": 0.9},
    {"content": "user plans to add a web interface in Phase 2", "category": "goal", "confidence": 0.7},
    {"content": "user wants to publish Jarvis as open source eventually", "category": "goal", "confidence": 0.6},
    {"content": "user's primary goal is to prove memory injection works in PoC", "category": "goal", "confidence": 0.95},
    {"content": "user wants to validate disinhibition retrieval beats cosine similarity", "category": "goal", "confidence": 0.85},
    {"content": "user wants to train a LoRA adapter on personal data by Month 2", "category": "goal", "confidence": 0.7},
    {"content": "user aims to have 250 beta users by Month 6", "category": "goal", "confidence": 0.65},
    {"content": "user wants to build a causal knowledge graph in Phase 3", "category": "goal", "confidence": 0.75},
    {"content": "user wants Jarvis accessible from Claude Desktop via MCP", "category": "goal", "confidence": 0.9},
    # ID 21-30: Expertise
    {"content": "user has 10 years of Python software engineering experience", "category": "expertise", "confidence": 0.9},
    {"content": "user is intermediate in machine learning concepts", "category": "expertise", "confidence": 0.75},
    {"content": "user has used networkx for graph algorithms before", "category": "expertise", "confidence": 0.7},
    {"content": "user knows TF-IDF and BM25 retrieval algorithms", "category": "expertise", "confidence": 0.8},
    {"content": "user has experience with SQLite WAL mode and transactions", "category": "expertise", "confidence": 0.75},
    {"content": "user has built REST APIs with Flask before", "category": "expertise", "confidence": 0.8},
    {"content": "user is learning about causal inference and DoWhy", "category": "expertise", "confidence": 0.6},
    {"content": "user is familiar with transformer attention mechanisms", "category": "expertise", "confidence": 0.65},
    {"content": "user has used LoRA fine-tuning on open source models", "category": "expertise", "confidence": 0.55},
    {"content": "user is familiar with Ollama API and local LLM deployment", "category": "expertise", "confidence": 0.9},
    # ID 31-40: Patterns
    {"content": "user does focused work in 2-hour morning blocks", "category": "pattern", "confidence": 0.8},
    {"content": "user reads technical papers in the evening", "category": "pattern", "confidence": 0.7},
    {"content": "user journals about work decisions when stuck", "category": "pattern", "confidence": 0.6},
    {"content": "user reviews code at end of day, not start", "category": "pattern", "confidence": 0.65},
    {"content": "user gets distracted by new tools and has to self-regulate", "category": "pattern", "confidence": 0.7},
    {"content": "user tracks tasks in Linear, not Notion", "category": "pattern", "confidence": 0.75},
    {"content": "user communicates more effectively in writing than speaking", "category": "pattern", "confidence": 0.65},
    {"content": "user works best with a defined scope before starting", "category": "pattern", "confidence": 0.8},
    {"content": "user often identifies the simplest solution after sleeping on problems", "category": "pattern", "confidence": 0.7},
    {"content": "user prefers finishing one task fully before starting the next", "category": "pattern", "confidence": 0.75},
    # ID 41-50: General context
    {"content": "user uses MacBook Pro M3 Pro with 36GB unified memory", "category": "general", "confidence": 0.95},
    {"content": "user has Ollama installed with llama3, zephyr, codegemma models", "category": "general", "confidence": 0.9},
    {"content": "user is in a city timezone (not specified)", "category": "general", "confidence": 0.5},
    {"content": "user's primary interface is terminal and Claude Desktop", "category": "general", "confidence": 0.85},
    {"content": "user has a GitHub account and keeps most projects public", "category": "general", "confidence": 0.7},
    {"content": "user uses Python virtual environments for all projects", "category": "general", "confidence": 0.8},
    {"content": "user writes markdown for all documentation", "category": "general", "confidence": 0.75},
    {"content": "user has tried ChatGPT Plus and Claude Pro before starting Jarvis", "category": "general", "confidence": 0.8},
    {"content": "user's laptop runs macOS Sequoia", "category": "general", "confidence": 0.85},
    {"content": "user has no external monitor, works on laptop display only", "category": "general", "confidence": 0.6},
]

# 20 queries + expected top-1 memory index (0-based)
QUERIES_WITH_ANSWERS = [
    ("What does the user prefer for database storage in local projects?", 4),   # SQLite over Postgres
    ("What is the user currently building?", 10),   # Project Jarvis
    ("What are the user's Python coding style preferences?", 1),   # type hints
    ("What hardware does the user work on?", 40),   # MacBook M3
    ("What's the user's primary goal for the PoC?", 14),   # prove memory injection
    ("How does the user like to receive answers?", 0),   # direct, no preamble
    ("What AI models does the user have locally?", 41),   # ollama models
    ("What tools does the user use to track tasks?", 35),   # Linear
    ("When does the user prefer to do focused work?", 30),   # morning blocks
    ("What retrieval algorithms does the user know?", 23),   # TF-IDF, BM25
    ("What is the user's experience level with Python?", 20),   # 10 years
    ("Does the user prefer sync or async code?", 9),   # sync for PoC
    ("What testing framework does the user use?", 3),   # pytest
    ("What does the user want Jarvis to eventually be?", 12),   # open source
    ("What interface does the user primarily work in?", 43),   # terminal + Claude Desktop
    ("How experienced is the user with local LLM deployment?", 29),   # Ollama familiar
    ("What causal reasoning tool is the user learning?", 26),   # DoWhy
    ("What is the user's MCP-related goal for Jarvis?", 19),   # Claude Desktop via MCP
    ("What does the user do when stuck on a problem?", 32),   # journals
    ("What's the user's approach to dependencies?", 2),   # minimal deps
]


def get_embedding(text: str) -> list[float] | None:
    """Get embedding from Ollama. Returns None if model not available."""
    import requests
    try:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception as e:
        print(f"  [embed] Failed: {e}")
        return None


def recall_at_k(results: list[dict], expected_id: int, k: int) -> bool:
    """Check if expected memory appears in top-k results (by DB id)."""
    top_ids = [r.get("id") for r in results[:k]]
    return expected_id in top_ids


def run_experiment():
    model = pick_model()
    embed_available = EMBED_MODEL in available_models()
    print(f"\n{'='*60}")
    print("Experiment 5 — Retrieval: TF-IDF vs. Embeddings")
    print(f"Model: {model}")
    print(f"Embedding model available ({EMBED_MODEL}): {embed_available}")
    if not embed_available:
        print(f"  → Method B will be skipped. Run: ollama pull {EMBED_MODEL}")
    print(f"{'='*60}\n")

    conn = fresh_db()
    ids = seed_memories(conn, MEMORIES_50)
    memories = get_memories(conn)
    print(f"Seeded {len(memories)} memories.\n")

    # Pre-compute embeddings for all memories if available
    memory_embeddings = {}
    if embed_available:
        print(f"Computing embeddings for {len(memories)} memories...")
        for m in memories:
            emb = get_embedding(m["content"])
            if emb:
                memory_embeddings[m["id"]] = emb
        print(f"  Embedded: {len(memory_embeddings)}/{len(memories)}")

    results_a, results_b = [], []

    for q_idx, (query, expected_mem_idx) in enumerate(QUERIES_WITH_ANSWERS, 1):
        expected_db_id = ids[expected_mem_idx]  # actual DB row id for this memory

        # Method A — TF-IDF
        top_a = retrieve_tfidf(query, memories, top_n=5)
        r1_a  = recall_at_k(top_a, expected_db_id, 1)
        r5_a  = recall_at_k(top_a, expected_db_id, 5)
        results_a.append({"query": query, "expected_id": expected_db_id,
                           "recall@1": r1_a, "recall@5": r5_a,
                           "top_ids": [m["id"] for m in top_a]})

        # Method B — Embeddings (skip if model not available)
        r1_b = r5_b = None
        if embed_available:
            q_emb = get_embedding(query)
            if q_emb:
                # Attach embeddings to memories for retrieval
                mems_with_emb = [
                    {**m, "embedding": memory_embeddings.get(m["id"], [])}
                    for m in memories
                ]
                from core.retrieval import retrieve_embeddings
                top_b = retrieve_embeddings(q_emb, mems_with_emb, top_n=5)
                r1_b  = recall_at_k(top_b, expected_db_id, 1)
                r5_b  = recall_at_k(top_b, expected_db_id, 5)
                results_b.append({"query": query, "expected_id": expected_db_id,
                                   "recall@1": r1_b, "recall@5": r5_b,
                                   "top_ids": [m["id"] for m in top_b]})

        status_a = "✓" if r1_a else ("~" if r5_a else "✗")
        status_b = ("✓" if r1_b else ("~" if r5_b else "✗")) if r1_b is not None else "-"
        print(f"  Q{q_idx:2d}: A={status_a} B={status_b} | {query[:55]}...")

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(QUERIES_WITH_ANSWERS)
    r1_a_pct = sum(r["recall@1"] for r in results_a) / n
    r5_a_pct = sum(r["recall@5"] for r in results_a) / n

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Method A — TF-IDF:    Recall@1 = {r1_a_pct:.0%} | Recall@5 = {r5_a_pct:.0%}")

    decision = ""
    if embed_available and results_b:
        r1_b_pct = sum(r["recall@1"] for r in results_b) / len(results_b)
        r5_b_pct = sum(r["recall@5"] for r in results_b) / len(results_b)
        print(f"Method B — Embeddings: Recall@1 = {r1_b_pct:.0%} | Recall@5 = {r5_b_pct:.0%}")
        if r5_a_pct >= 0.75:
            decision = f"PASS — TF-IDF Recall@5 = {r5_a_pct:.0%} ≥ 75%. Use TF-IDF for PoC."
        else:
            decision = f"Use embeddings — TF-IDF Recall@5 = {r5_a_pct:.0%} < 75%. " \
                       f"Embeddings give {r5_b_pct:.0%}."
    else:
        if r5_a_pct >= 0.75:
            decision = f"PASS — TF-IDF Recall@5 = {r5_a_pct:.0%} ≥ 75%. Use TF-IDF for PoC."
        else:
            decision = f"TF-IDF Recall@5 = {r5_a_pct:.0%} < 75%. Consider pulling nomic-embed-text."

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":        "exp_05_retrieval",
        "model":             model,
        "embed_available":   embed_available,
        "run_at":            ts,
        "tfidf_recall1":     r1_a_pct,
        "tfidf_recall5":     r5_a_pct,
        "decision":          decision,
        "results_tfidf":     results_a,
        "results_embed":     results_b if embed_available else [],
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 5 — Retrieval\nRun: {ts}\n"
        f"TF-IDF: Recall@1={r1_a_pct:.0%} Recall@5={r5_a_pct:.0%}\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
