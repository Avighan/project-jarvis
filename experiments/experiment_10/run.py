"""
Experiment 10 — Auto-Contradiction Detection
=============================================
Question: Does auto-contradiction detection correctly identify conflicting
memories and create contradicts edges?

Method:
- Seed 10 base memories (high confidence, 0.85-0.95)
- Attempt to add 10 new memories: 5 contradictions + 5 compatible
- Auto-detection: TF-IDF cosine similarity > 0.4 AND category match AND
  Ollama "Does B contradict A? YES/NO" check
- For confirmed contradictions: create contradicts edge + update_confidence(-0.15)

Pass criterion: true_positives >= 4 AND false_positives == 0

Run: python3 POC/experiments/experiment_10/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama  import generate, pick_model
from experiments.shared.db      import fresh_db, seed_memories, get_memories
from core.retrieval             import build_idf, tfidf_score

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Seed memories (10, high confidence) ──────────────────────────────────────

SEED_MEMORIES = [
    {"content": "user prefers direct responses without preamble",           "category": "preference", "confidence": 0.9},
    {"content": "user works in 2-hour morning focus blocks",                "category": "pattern",    "confidence": 0.85},
    {"content": "user prefers synchronous Python code for PoC",             "category": "preference", "confidence": 0.85},
    {"content": "user tracks tasks in Linear",                              "category": "pattern",    "confidence": 0.85},
    {"content": "user dislikes over-explained answers",                     "category": "preference", "confidence": 0.9},
    {"content": "user is building Project Jarvis in Python 3.12",           "category": "goal",       "confidence": 0.95},
    {"content": "user prefers minimal dependencies in Python projects",     "category": "preference", "confidence": 0.85},
    {"content": "user has 10 years of Python experience",                   "category": "expertise",  "confidence": 0.9},
    {"content": "user finds context switching disruptive to deep work",     "category": "pattern",    "confidence": 0.8},
    {"content": "user uses VS Code as primary editor",                      "category": "general",    "confidence": 0.75},
]

# ── New memories to test ──────────────────────────────────────────────────────
# Each has: content, category, confidence, expected_contradiction (bool),
#           contradicts_seed_index (0-based index into SEED_MEMORIES, or None)

NEW_MEMORIES = [
    # ── CONTRADICTIONS (5) — should be detected ───────────────────────────────
    {
        "content":                   "user wants verbose detailed explanations with full context",
        "category":                  "preference",
        "confidence":                0.7,
        "expected_contradiction":    True,
        "contradicts_seed_index":    4,   # "user dislikes over-explained answers"
        "label":                     "C1",
    },
    {
        "content":                   "user prefers working late at night, not mornings",
        "category":                  "pattern",
        "confidence":                0.7,
        "expected_contradiction":    True,
        "contradicts_seed_index":    1,   # "user works in 2-hour morning focus blocks"
        "label":                     "C2",
    },
    {
        "content":                   "user now prefers async Python after switching to FastAPI",
        "category":                  "preference",
        "confidence":                0.7,
        "expected_contradiction":    True,
        "contradicts_seed_index":    2,   # "user prefers synchronous Python code for PoC"
        "label":                     "C3",
    },
    {
        "content":                   "user switched from Linear to Notion for task tracking",
        "category":                  "pattern",
        "confidence":                0.7,
        "expected_contradiction":    True,
        "contradicts_seed_index":    3,   # "user tracks tasks in Linear"
        "label":                     "C4",
    },
    {
        "content":                   "user is fine with adding many dependencies if it saves time",
        "category":                  "preference",
        "confidence":                0.7,
        "expected_contradiction":    True,
        "contradicts_seed_index":    6,   # "user prefers minimal dependencies in Python projects"
        "label":                     "C5",
    },
    # ── COMPATIBLE (5) — should NOT be flagged ────────────────────────────────
    {
        "content":                   "user reads technical papers on weekend mornings",
        "category":                  "pattern",
        "confidence":                0.7,
        "expected_contradiction":    False,
        "contradicts_seed_index":    None,
        "label":                     "P1",
    },
    {
        "content":                   "user has used BM25 in a previous Java project",
        "category":                  "expertise",
        "confidence":                0.65,
        "expected_contradiction":    False,
        "contradicts_seed_index":    None,
        "label":                     "P2",
    },
    {
        "content":                   "user is interested in building a SaaS version of Jarvis",
        "category":                  "goal",
        "confidence":                0.6,
        "expected_contradiction":    False,
        "contradicts_seed_index":    None,
        "label":                     "P3",
    },
    {
        "content":                   "user prefers writing integration tests over unit tests",
        "category":                  "preference",
        "confidence":                0.65,
        "expected_contradiction":    False,
        "contradicts_seed_index":    None,
        "label":                     "P4",
    },
    {
        "content":                   "user uses GitHub Actions for CI on some projects",
        "category":                  "general",
        "confidence":                0.6,
        "expected_contradiction":    False,
        "contradicts_seed_index":    None,
        "label":                     "P5",
    },
]

SIMILARITY_THRESHOLD = 0.4


def tfidf_similarity(text_a: str, text_b: str, corpus: list[str]) -> float:
    """
    Compute a symmetric TF-IDF similarity between two texts using the
    shared corpus IDF. Uses the average of score(a→b) and score(b→a)
    normalised by each document's self-score (to bound to [0, 1]).
    """
    idf = build_idf(corpus + [text_a, text_b])
    score_ab = tfidf_score(text_a, text_b, idf)
    score_ba = tfidf_score(text_b, text_a, idf)
    self_a   = tfidf_score(text_a, text_a, idf)
    self_b   = tfidf_score(text_b, text_b, idf)
    norm_a   = score_ab / self_a if self_a > 0 else 0.0
    norm_b   = score_ba / self_b if self_b > 0 else 0.0
    return (norm_a + norm_b) / 2.0


def llm_contradiction_check(existing: str, new_mem: str, model: str) -> bool:
    """Ask Ollama whether new_mem contradicts existing. Returns True if YES."""
    prompt = (
        f"Does statement B contradict statement A? "
        f"Answer YES or NO only.\n"
        f"A: {existing}\n"
        f"B: {new_mem}"
    )
    raw, _ = generate(prompt, model=model, temperature=0.0, max_tokens=5)
    return raw.strip().upper().startswith("YES")


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 10 — Auto-Contradiction Detection")
    print(f"Model: {model} | 10 seed + 10 new memories")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"{'='*60}\n")

    # ── Seed base memories ────────────────────────────────────────────────────
    conn = fresh_db()
    seed_ids = seed_memories(conn, SEED_MEMORIES)
    base_memories = get_memories(conn)
    print(f"Seeded {len(base_memories)} base memories (IDs: {seed_ids[0]}–{seed_ids[-1]})\n")

    corpus = [m["content"] for m in base_memories]

    all_results         = []
    true_positives      = 0   # contradictions correctly detected
    false_positives     = 0   # compatible pairs wrongly flagged
    true_negatives      = 0   # compatible pairs correctly left alone
    false_negatives     = 0   # contradictions missed
    edges_created       = []
    confidence_updates  = []

    for new_mem in NEW_MEMORIES:
        new_content  = new_mem["content"]
        new_category = new_mem["category"]
        expected     = new_mem["expected_contradiction"]
        label        = new_mem["label"]

        print(f"\n[{label}] {new_content[:70]}...")
        print(f"      Expected contradiction: {expected}")

        # Step 1: Insert the new memory so we can reference its id later
        cur = conn.execute(
            "INSERT INTO memories (content, category, confidence) VALUES (?,?,?)",
            (new_content, new_category, new_mem["confidence"])
        )
        conn.commit()
        new_id = cur.lastrowid

        # Step 2: Compare against all existing seed memories
        detected_contradiction = False
        detected_against_id    = None
        detected_against_text  = None
        similarity_score       = 0.0
        llm_said_yes           = False

        for existing in base_memories:
            # Gate 1: category match
            if existing["category"] != new_category:
                continue

            # Gate 2: TF-IDF similarity
            sim = tfidf_similarity(new_content, existing["content"], corpus)
            if sim <= SIMILARITY_THRESHOLD:
                continue

            # Gate 3: LLM semantic check
            llm_yes = llm_contradiction_check(existing["content"], new_content, model)

            print(f"      ↳ vs '{existing['content'][:55]}...'")
            print(f"        similarity={sim:.3f}  llm_yes={llm_yes}")

            if llm_yes:
                detected_contradiction = True
                detected_against_id    = existing["id"]
                detected_against_text  = existing["content"]
                similarity_score       = sim
                llm_said_yes           = True
                break  # take the first confirmed contradiction

        # Step 3: If contradiction confirmed, create edge + update confidence
        edge_created      = False
        confidence_updated = False
        old_confidence     = None
        new_confidence     = None

        if detected_contradiction and detected_against_id is not None:
            # Create contradicts edge (new memory → existing memory)
            conn.execute(
                "INSERT INTO memory_edges (source_id, target_id, edge_type, weight) VALUES (?,?,?,?)",
                (new_id, detected_against_id, "contradicts", 1.0)
            )
            conn.commit()
            edge_created = True
            edges_created.append((new_id, detected_against_id))

            # Fetch old confidence before update
            row = conn.execute(
                "SELECT confidence FROM memories WHERE id=?", (detected_against_id,)
            ).fetchone()
            old_confidence = row["confidence"] if row else None

            # Apply confidence penalty directly on in-memory DB
            conn.execute(
                "UPDATE memories SET confidence = MAX(0.0, MIN(1.0, confidence - 0.15)) WHERE id=?",
                (detected_against_id,)
            )
            conn.commit()

            row2 = conn.execute(
                "SELECT confidence FROM memories WHERE id=?", (detected_against_id,)
            ).fetchone()
            new_confidence = row2["confidence"] if row2 else None
            confidence_updated = (
                old_confidence is not None
                and new_confidence is not None
                and abs((old_confidence - 0.15) - new_confidence) < 0.001
            )
            confidence_updates.append({
                "memory_id":       detected_against_id,
                "old_confidence":  old_confidence,
                "new_confidence":  new_confidence,
                "updated_correct": confidence_updated,
            })

        # Scoring
        if expected and detected_contradiction:
            true_positives += 1
            outcome = "TP"
        elif expected and not detected_contradiction:
            false_negatives += 1
            outcome = "FN"
        elif not expected and detected_contradiction:
            false_positives += 1
            outcome = "FP"
        else:
            true_negatives += 1
            outcome = "TN"

        print(f"      Detected: {detected_contradiction}  Outcome: {outcome}  "
              f"edge_created: {edge_created}  confidence_updated: {confidence_updated}")

        all_results.append({
            "label":               label,
            "new_content":         new_content,
            "new_category":        new_category,
            "expected":            expected,
            "detected":            detected_contradiction,
            "detected_against":    detected_against_text,
            "similarity_score":    round(similarity_score, 4),
            "llm_said_yes":        llm_said_yes,
            "outcome":             outcome,
            "edge_created":        edge_created,
            "old_confidence":      old_confidence,
            "new_confidence":      new_confidence,
            "confidence_updated":  confidence_updated,
        })

    # ── Verify edges in DB ────────────────────────────────────────────────────
    edge_rows = conn.execute(
        "SELECT * FROM memory_edges WHERE edge_type='contradicts'"
    ).fetchall()
    edges_in_db = len(edge_rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"True positives  (contradictions caught):      {true_positives}/5")
    print(f"False positives (compatible wrongly flagged): {false_positives}/5")
    print(f"False negatives (contradictions missed):      {false_negatives}/5")
    print(f"True negatives  (compatible correctly passed):{true_negatives}/5")
    print(f"Contradicts edges in DB:                      {edges_in_db}")
    conf_correct = sum(1 for u in confidence_updates if u["updated_correct"])
    print(f"Confidence updates correct:                   {conf_correct}/{len(confidence_updates)}")

    # Decision
    if true_positives >= 4 and false_positives == 0:
        decision = "PASS — auto-contradiction detection works; build into write path"
    elif true_positives >= 3 or false_positives <= 1:
        decision = "PARTIAL — detection works but needs threshold tuning"
    else:
        decision = "FAIL — too many false positives or misses; use manual-only contradiction marking"

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":          "exp_10_contradiction_detection",
        "model":               model,
        "run_at":              ts,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "true_positives":      true_positives,
        "false_positives":     false_positives,
        "false_negatives":     false_negatives,
        "true_negatives":      true_negatives,
        "edges_in_db":         edges_in_db,
        "confidence_updates":  confidence_updates,
        "pass_criterion":      "true_positives >= 4 AND false_positives == 0",
        "decision":            decision,
        "results":             all_results,
    }

    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))

    txt_path = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 10 — Auto-Contradiction Detection\n"
        f"Run: {ts} | Model: {model}\n"
        f"Threshold: {SIMILARITY_THRESHOLD}\n"
        f"True positives: {true_positives}/5 | False positives: {false_positives}/5\n"
        f"False negatives: {false_negatives}/5 | True negatives: {true_negatives}/5\n"
        f"Edges in DB: {edges_in_db}\n"
        f"Decision: {decision}\n"
    )

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
