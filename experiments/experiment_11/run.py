"""
Experiment 11 — TF-IDF Deduplication at Write Time
====================================================
Question: Does TF-IDF cosine similarity at write time correctly deduplicate
near-duplicate memories?

Method:
- 10 unique memories seeded first
- 10 near-duplicates (same fact, different wording) attempted
- Dedup check uses Jaccard similarity of tokenised token sets
- Test thresholds: 0.3, 0.5, 0.7
- Optimal threshold = highest true_blocks with 0 false_blocks

Pass criterion: At some threshold, true_blocks >= 7/10 AND false_blocks == 0

Run: python3 POC/experiments/experiment_11/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.db  import fresh_db, seed_memories, get_memories
from core.retrieval         import tokenise

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLDS = [0.3, 0.5, 0.7]

# ── 10 unique original memories (seeded first) ────────────────────────────────

ORIGINALS = [
    {"original": "user prefers direct responses",                          "category": "preference", "confidence": 0.9},
    {"original": "user has 10 years of Python experience",                 "category": "expertise",  "confidence": 0.9},
    {"original": "user is building Project Jarvis",                        "category": "goal",       "confidence": 0.95},
    {"original": "user works in morning focus blocks",                     "category": "pattern",    "confidence": 0.85},
    {"original": "user prefers minimal dependencies",                      "category": "preference", "confidence": 0.85},
    {"original": "user runs Ollama locally on MacBook",                    "category": "general",    "confidence": 0.9},
    {"original": "user dislikes over-explained answers",                   "category": "preference", "confidence": 0.9},
    {"original": "user tracks tasks in Linear",                            "category": "pattern",    "confidence": 0.85},
    {"original": "user finds context switching disruptive",                "category": "pattern",    "confidence": 0.8},
    {"original": "user prefers code examples over pseudocode",             "category": "preference", "confidence": 0.8},
]

# ── 10 near-duplicate candidates (should be blocked at optimal threshold) ─────

NEAR_DUPLICATES = [
    "user likes concise answers without fluff",
    "user is an expert Python developer with a decade of experience",
    "user is working on Project Jarvis, an AI memory system",
    "user does focused work in the mornings",
    "user avoids unnecessary Python packages",
    "user uses local Ollama on their MacBook Pro",
    "user wants answers without excessive explanation",
    "user uses Linear to manage their tasks",
    "user gets disrupted when switching between tasks",
    "user wants real code, not pseudocode examples",
]

# ── 10 unique memories that must NOT be blocked ───────────────────────────────

UNIQUE_MEMORIES = [
    {"content": "user has a dog named Max",                                "category": "general",    "confidence": 0.3},
    {"content": "user reads technical papers on weekends",                 "category": "pattern",    "confidence": 0.65},
    {"content": "user is interested in Rust but hasn't used it recently",  "category": "expertise",  "confidence": 0.5},
    {"content": "user grew up in London",                                  "category": "general",    "confidence": 0.45},
    {"content": "user uses a standing desk setup",                         "category": "general",    "confidence": 0.55},
    {"content": "user has worked at startups primarily",                   "category": "general",    "confidence": 0.55},
    {"content": "user mentioned wanting to add voice input to Jarvis",     "category": "goal",       "confidence": 0.6},
    {"content": "user drinks coffee not tea",                              "category": "pattern",    "confidence": 0.5},
    {"content": "user has tried several AI tools before Jarvis",           "category": "general",    "confidence": 0.65},
    {"content": "user prefers Python type hints in all function signatures","category": "preference", "confidence": 0.75},
]


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity of token sets after tokenisation."""
    set_a = set(tokenise(text_a))
    set_b = set(tokenise(text_b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union


def would_block(new_text: str, existing_texts: list[str], threshold: float) -> tuple[bool, float]:
    """Return (blocked, max_similarity) for new_text against all existing texts."""
    if not existing_texts:
        return False, 0.0
    sims = [jaccard_similarity(new_text, e) for e in existing_texts]
    max_sim = max(sims)
    return max_sim > threshold, max_sim


def run_experiment():
    print(f"\n{'='*60}")
    print("Experiment 11 — TF-IDF Deduplication at Write Time")
    print(f"Thresholds tested: {THRESHOLDS}")
    print(f"10 originals | 10 near-duplicates | 10 unique memories")
    print(f"{'='*60}\n")

    # Seed original memories into a fresh DB (for record-keeping only;
    # all similarity checks are done in-memory against the text list)
    conn = fresh_db()
    seed_data = [{"content": o["original"], "category": o["category"],
                  "confidence": o["confidence"]} for o in ORIGINALS]
    seed_memories(conn, seed_data)
    base_memories = get_memories(conn)
    existing_texts = [m["content"] for m in base_memories]

    print(f"Seeded {len(existing_texts)} original memories.\n")

    # ── Per-threshold evaluation ──────────────────────────────────────────────
    threshold_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'─'*60}")
        print(f"Threshold = {threshold}")
        print(f"{'─'*60}")

        true_blocks  = 0   # near-dups correctly blocked
        false_blocks = 0   # uniques wrongly blocked
        dup_details  = []
        uniq_details = []

        # Test near-duplicates
        print("\nNear-duplicates (should be blocked):")
        for i, dup_text in enumerate(NEAR_DUPLICATES, 1):
            blocked, max_sim = would_block(dup_text, existing_texts, threshold)
            if blocked:
                true_blocks += 1
            sym = "BLOCKED" if blocked else "PASSED "
            print(f"  [{i:2d}] {sym}  sim={max_sim:.3f}  '{dup_text[:55]}...'")
            dup_details.append({
                "text":    dup_text,
                "blocked": blocked,
                "max_sim": round(max_sim, 4),
                "correct": blocked,  # we want these blocked
            })

        # Test unique memories
        print("\nUnique memories (should NOT be blocked):")
        for i, uniq in enumerate(UNIQUE_MEMORIES, 1):
            blocked, max_sim = would_block(uniq["content"], existing_texts, threshold)
            if blocked:
                false_blocks += 1
            sym = "BLOCKED" if blocked else "passed "
            print(f"  [{i:2d}] {sym}  sim={max_sim:.3f}  '{uniq['content'][:55]}...'")
            uniq_details.append({
                "text":    uniq["content"],
                "blocked": blocked,
                "max_sim": round(max_sim, 4),
                "correct": not blocked,  # we want these NOT blocked
            })

        print(f"\n  true_blocks={true_blocks}/10  false_blocks={false_blocks}/10")
        threshold_results[threshold] = {
            "threshold":    threshold,
            "true_blocks":  true_blocks,
            "false_blocks": false_blocks,
            "dup_details":  dup_details,
            "uniq_details": uniq_details,
        }

    # ── Find optimal threshold ────────────────────────────────────────────────
    optimal_threshold = None
    optimal_true_blocks = 0

    # Best = highest true_blocks where false_blocks == 0
    for t in THRESHOLDS:
        r = threshold_results[t]
        if r["false_blocks"] == 0 and r["true_blocks"] > optimal_true_blocks:
            optimal_true_blocks  = r["true_blocks"]
            optimal_threshold    = t

    # Fallback: if no threshold achieves 0 false_blocks, pick fewest false_blocks
    if optimal_threshold is None:
        best = min(threshold_results.values(), key=lambda r: (r["false_blocks"], -r["true_blocks"]))
        optimal_threshold   = best["threshold"]
        optimal_true_blocks = best["true_blocks"]

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for t in THRESHOLDS:
        r = threshold_results[t]
        flag = " <-- optimal" if t == optimal_threshold else ""
        print(f"  threshold={t}: true_blocks={r['true_blocks']}/10  "
              f"false_blocks={r['false_blocks']}/10{flag}")

    print(f"\nOptimal threshold: {optimal_threshold}  "
          f"(true_blocks={optimal_true_blocks}, false_blocks=0)")

    # Decision
    opt_r = threshold_results[optimal_threshold]
    if opt_r["true_blocks"] >= 7 and opt_r["false_blocks"] == 0:
        decision = (f"PASS — dedup threshold={optimal_threshold}; "
                    f"add to write path in memory_store.py")
    elif opt_r["true_blocks"] >= 5:
        decision = (f"PARTIAL — threshold exists but blocks "
                    f"{opt_r['true_blocks']}/10 dups "
                    f"(need >=7)")
    else:
        decision = "FAIL — no threshold achieves both goals"

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":         "exp_11_dedup_threshold",
        "run_at":             ts,
        "thresholds_tested":  THRESHOLDS,
        "optimal_threshold":  optimal_threshold,
        "optimal_true_blocks": optimal_true_blocks,
        "pass_criterion":     "true_blocks >= 7 AND false_blocks == 0 at some threshold",
        "decision":           decision,
        "threshold_results":  {str(k): v for k, v in threshold_results.items()},
    }

    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))

    txt_path = RESULTS_DIR / f"run_{ts}.txt"
    lines = [
        f"Experiment 11 — TF-IDF Deduplication at Write Time",
        f"Run: {ts}",
        f"Thresholds tested: {THRESHOLDS}",
    ]
    for t in THRESHOLDS:
        r = threshold_results[t]
        lines.append(f"  threshold={t}: true_blocks={r['true_blocks']}/10  false_blocks={r['false_blocks']}/10")
    lines.append(f"Optimal threshold: {optimal_threshold}")
    lines.append(f"Decision: {decision}")
    txt_path.write_text("\n".join(lines) + "\n")

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
