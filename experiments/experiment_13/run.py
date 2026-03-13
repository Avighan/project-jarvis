"""
Experiment 13 — CausalRAG: Does Grounding Causal Answers in a Personal Causal Graph
                            Improve Accuracy vs LLM Alone?
=====================================================================================
Tests the CausalRAG concept from the architecture spec (Section 4.7).

Method A: LLM receives only the bare question (no causal context)
Method B: LLM receives relevant causal edges injected before the question

Pass: avg_b > avg_a by >= 0.5 points AND method_b_wins >= 5/8

Run: python3 POC/experiments/experiment_13/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from experiments.shared.judge  import rate
from core.working_memory        import build_prompt_no_memory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Personal causal graph (Python dicts — NOT stored in DB for this experiment) ──
CAUSAL_GRAPH = [
    {
        "cause":      "poor sleep (< 6 hours)",
        "effect":     "low focus score next morning",
        "lag_hours":  8,
        "strength":   0.85,
        "confidence": 0.90,
        "rung":       2,
        "domain":     "health",
    },
    {
        "cause":      "high meeting density (> 4 meetings/day)",
        "effect":     "reduced deep work output",
        "lag_hours":  0,
        "strength":   0.80,
        "confidence": 0.85,
        "rung":       2,
        "domain":     "productivity",
    },
    {
        "cause":      "morning exercise",
        "effect":     "improved focus for 3-4 hours",
        "lag_hours":  1,
        "strength":   0.75,
        "confidence": 0.80,
        "rung":       2,
        "domain":     "health",
    },
    {
        "cause":      "context switching between projects",
        "effect":     "reduced code quality + increased bugs",
        "lag_hours":  0,
        "strength":   0.78,
        "confidence": 0.82,
        "rung":       2,
        "domain":     "productivity",
    },
    {
        "cause":      "working past 10pm",
        "effect":     "lower productivity next morning",
        "lag_hours":  10,
        "strength":   0.70,
        "confidence": 0.75,
        "rung":       2,
        "domain":     "health",
    },
    {
        "cause":      "journaling before coding",
        "effect":     "clearer architecture decisions",
        "lag_hours":  0,
        "strength":   0.65,
        "confidence": 0.70,
        "rung":       2,
        "domain":     "productivity",
    },
    {
        "cause":      "coffee after 3pm",
        "effect":     "disrupted sleep",
        "lag_hours":  6,
        "strength":   0.72,
        "confidence": 0.78,
        "rung":       2,
        "domain":     "health",
    },
    {
        "cause":      "skipping lunch break",
        "effect":     "afternoon energy crash",
        "lag_hours":  2,
        "strength":   0.68,
        "confidence": 0.72,
        "rung":       2,
        "domain":     "health",
    },
]

# ── 8 test questions (4 causal "why" + 4 intervention "what should I do") ────
TEST_QUESTIONS = [
    # CAUSAL — why
    {
        "type": "causal",
        "question": "Why am I often tired and unfocused in the morning after late nights?",
    },
    {
        "type": "causal",
        "question": "Why does my code quality drop when I have back-to-back meetings?",
    },
    {
        "type": "causal",
        "question": "Why do I feel foggy after days with lots of context switching?",
    },
    {
        "type": "causal",
        "question": "What is causing my afternoon energy crashes?",
    },
    # INTERVENTION — what should I do
    {
        "type": "intervention",
        "question": "What should I do the night before an important coding day?",
    },
    {
        "type": "intervention",
        "question": "How should I structure my calendar to maximise deep work output?",
    },
    {
        "type": "intervention",
        "question": "What morning habits would most improve my focus for coding?",
    },
    {
        "type": "intervention",
        "question": "How can I improve my architecture decision-making process?",
    },
]


def retrieve_relevant_edges(question: str) -> list[dict]:
    """
    Simple keyword overlap retrieval: if any word from the question
    (> 4 chars) appears in cause or effect string (case-insensitive),
    include that edge.
    """
    q_words = {w.lower() for w in question.split() if len(w) > 4}
    relevant = []
    for edge in CAUSAL_GRAPH:
        cause_lower  = edge["cause"].lower()
        effect_lower = edge["effect"].lower()
        if any(w in cause_lower or w in effect_lower for w in q_words):
            relevant.append(edge)
    return relevant


def format_causal_context(edges: list[dict]) -> str:
    """Format causal edges for injection into Method B prompt."""
    if not edges:
        return ""
    lines = ["Your personal causal knowledge (verified from your own data):"]
    for e in edges:
        lines.append(
            f"- {e['cause']} → {e['effect']} "
            f"(strength: {e['strength']:.0%}, lag: {e['lag_hours']}h, "
            f"confidence: {e['confidence']:.0%})"
        )
    return "\n".join(lines)


def run_experiment() -> None:
    model = pick_model()
    print(f"\n{'='*65}")
    print("Experiment 13 — CausalRAG: Personal Causal Graph vs LLM Alone")
    print(f"Model: {model} | 8 questions | Causal graph: {len(CAUSAL_GRAPH)} edges")
    print(f"{'='*65}\n")

    # fresh_db() for isolation (no memories needed for this experiment)
    conn = fresh_db()
    # Seed a small set of general user memories to give the judge context
    base_memories = [
        {"content": "user is building Project Jarvis in Python 3.12",    "category": "goal",      "confidence": 0.95},
        {"content": "user works in 2-hour morning focus blocks",          "category": "pattern",   "confidence": 0.85},
        {"content": "user finds context switching disruptive to deep work","category": "pattern",   "confidence": 0.80},
        {"content": "user has 10 years Python experience; expert level",  "category": "expertise", "confidence": 0.92},
        {"content": "user tracks tasks in Linear",                        "category": "pattern",   "confidence": 0.82},
    ]
    seed_memories(conn, base_memories)
    memories = get_memories(conn)
    mem_contents = [m["content"] for m in memories]

    print(f"Seeded {len(memories)} base context memories for judge.\n")
    print(f"Causal graph edges:")
    for e in CAUSAL_GRAPH:
        print(f"  {e['cause'][:40]:40s} → {e['effect'][:40]}")
    print()

    all_results = []
    total_score_a = 0
    total_score_b = 0
    b_wins = 0

    for qi, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        q_type   = test["type"]
        print(f"Q{qi} [{q_type.upper()}]: {question[:60]}...")

        # Retrieve relevant causal edges
        relevant_edges = retrieve_relevant_edges(question)
        print(f"  Relevant edges matched: {len(relevant_edges)}/{len(CAUSAL_GRAPH)}")

        # Method A: bare question (no causal context)
        prompt_a = build_prompt_no_memory(question)
        resp_a, lat_a = generate(prompt_a, model=model, max_tokens=350)

        # Method B: causal context injected
        causal_ctx = format_causal_context(relevant_edges)
        if causal_ctx:
            prompt_b = causal_ctx + "\n\n" + question
        else:
            # Fallback: use full causal graph if no keyword match
            prompt_b = format_causal_context(CAUSAL_GRAPH) + "\n\n" + question

        resp_b, lat_b = generate(prompt_b, model=model, max_tokens=350)

        # Rate both 1-5 using LLM judge (no memories needed per spec — pass base context)
        score_a = rate(question, resp_a, memories=mem_contents, model=model)
        score_b = rate(question, resp_b, memories=mem_contents, model=model)

        total_score_a += score_a
        total_score_b += score_b

        winner = "B" if score_b > score_a else ("A" if score_a > score_b else "tie")
        if winner == "B":
            b_wins += 1

        # Check: does Method B's answer reference the specific causal relationship?
        # Simple heuristic: does resp_b contain keywords from a matched edge?
        b_cites_causality = False
        for edge in relevant_edges:
            cause_kw  = edge["cause"].split()[:3]
            effect_kw = edge["effect"].split()[:3]
            if any(w.lower() in resp_b.lower() for w in cause_kw + effect_kw if len(w) > 3):
                b_cites_causality = True
                break

        print(f"  Scores — A:{score_a}  B:{score_b}  winner={winner}  B_cites_causality={b_cites_causality}")
        print(f"  Latency — A:{lat_a}ms  B:{lat_b}ms")

        all_results.append({
            "query_id":            qi,
            "type":                q_type,
            "question":            question,
            "relevant_edges":      len(relevant_edges),
            "b_cites_causality":   b_cites_causality,
            "method_a": {
                "response":   resp_a,
                "latency_ms": lat_a,
                "score":      score_a,
            },
            "method_b": {
                "response":      resp_b,
                "latency_ms":    lat_b,
                "score":         score_b,
                "edges_injected": [f"{e['cause']} → {e['effect']}" for e in relevant_edges],
            },
            "winner": winner,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    avg_a = total_score_a / len(TEST_QUESTIONS)
    avg_b = total_score_b / len(TEST_QUESTIONS)
    delta = avg_b - avg_a

    causal_results      = [r for r in all_results if r["type"] == "causal"]
    intervention_results = [r for r in all_results if r["type"] == "intervention"]
    b_wins_causal       = sum(1 for r in causal_results if r["winner"] == "B")
    b_wins_intervention = sum(1 for r in intervention_results if r["winner"] == "B")
    b_cites_count       = sum(1 for r in all_results if r["b_cites_causality"])

    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"Method B wins: {b_wins}/8  (causal: {b_wins_causal}/4, intervention: {b_wins_intervention}/4)")
    print(f"Avg score A:   {avg_a:.2f}")
    print(f"Avg score B:   {avg_b:.2f}  (delta: {delta:+.2f})")
    print(f"B cites specific causal relationship: {b_cites_count}/8")

    if delta >= 0.5 and b_wins >= 5:
        decision = (
            f"PASS — CausalRAG improves causal answers "
            f"(avg_B={avg_b:.2f} vs avg_A={avg_a:.2f}, delta={delta:+.2f}, B_wins={b_wins}/8); "
            f"build Personal Causal Graph in Phase 3"
        )
    elif delta > 0 or b_wins >= 4:
        decision = (
            f"PARTIAL — marginal improvement "
            f"(avg_B={avg_b:.2f} vs avg_A={avg_a:.2f}, delta={delta:+.2f}, B_wins={b_wins}/8); "
            f"validate with larger graph"
        )
    else:
        decision = (
            f"FAIL — LLM ignores causal context "
            f"(avg_B={avg_b:.2f} vs avg_A={avg_a:.2f}, delta={delta:+.2f}, B_wins={b_wins}/8)"
        )

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":          "exp_13_causal_rag",
        "model":               model,
        "run_at":              ts,
        "causal_graph_edges":  len(CAUSAL_GRAPH),
        "test_questions":      len(TEST_QUESTIONS),
        "b_wins":              b_wins,
        "b_wins_causal":       b_wins_causal,
        "b_wins_intervention": b_wins_intervention,
        "avg_score_a":         round(avg_a, 3),
        "avg_score_b":         round(avg_b, 3),
        "delta":               round(delta, 3),
        "b_cites_causality":   b_cites_count,
        "decision":            decision,
        "results":             all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 13 — CausalRAG\n"
        f"Run: {ts} | Model: {model}\n"
        f"B wins: {b_wins}/8 (causal: {b_wins_causal}/4, intervention: {b_wins_intervention}/4)\n"
        f"Avg score A: {avg_a:.2f} | Avg score B: {avg_b:.2f} | Delta: {delta:+.2f}\n"
        f"B cites causality: {b_cites_count}/8\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
