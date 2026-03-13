"""
Experiment 1 — Does Memory Injection Matter At All?
====================================================
The foundational question. Run before anything else.

Test A: Prompt → Ollama (no memory injected)
Test B: Prompt → Ollama (top-4 memories injected)

Pass criterion: Test B responses are rated higher than Test A on ≥7/10 prompts.
Run: python3 POC/experiments/experiment_01/run.py
"""

import sys, json, time, pathlib, datetime, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from core.retrieval             import retrieve_tfidf
from core.working_memory        import build_prompt, build_prompt_no_memory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Test data ─────────────────────────────────────────────────────────────────

SEED_FACTS = [
    {"content": "user prefers direct responses without preamble or filler phrases",
     "category": "preference", "confidence": 0.9},
    {"content": "user is building Project Jarvis — a persistent AI memory layer in Python 3.12",
     "category": "goal", "confidence": 0.95},
    {"content": "user has 10 years of software engineering experience; expert-level Python",
     "category": "expertise", "confidence": 0.9},
    {"content": "user is running Ollama locally on MacBook Pro M3 Pro with 36GB unified memory",
     "category": "general", "confidence": 0.85},
    {"content": "user dislikes over-explained answers — wants the answer, not the lecture",
     "category": "preference", "confidence": 0.9},
]

TEST_PROMPTS = [
    "How should I structure a Python module that handles SQLite connections?",
    "What's the quickest way to think about this architecture: local SQLite vs. networkx graph for storing memories?",
    "Give me a quick take on whether I should use async or sync for my Ollama calls.",
    "What's a good way to do keyword retrieval without installing extra packages?",
    "Should I build the memory graph in Phase 1 or wait until I have more data?",
    "How do I decide the right context window size for injecting memories?",
    "What's the simplest way to test if memory injection actually helps?",
    "How would you approach debugging a slow Ollama response?",
    "What's a reasonable schema for storing user preferences in SQLite?",
    "How should I handle the case where Ollama is not running when Jarvis starts?",
]

# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print(f"Experiment 1 — Memory Injection: Does It Matter?")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    # Setup in-memory DB with seed facts
    conn = fresh_db()
    seed_memories(conn, SEED_FACTS)
    memories = get_memories(conn)
    print(f"Seeded {len(memories)} memories into test database.\n")

    results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"Prompt {i:2d}/{len(TEST_PROMPTS)}: {prompt[:60]}...")

        # ── Test A: No memory ────────────────────────────────────────────────
        prompt_a = build_prompt_no_memory(prompt)
        response_a, latency_a = generate(prompt_a, model=model)

        # ── Test B: With top-4 memories ──────────────────────────────────────
        retrieved  = retrieve_tfidf(prompt, memories, top_n=4)
        prompt_b   = build_prompt(prompt, retrieved, fmt="structured")
        response_b, latency_b = generate(prompt_b, model=model)

        result = {
            "prompt_id":     i,
            "prompt":        prompt,
            "test_a": {
                "response":    response_a,
                "latency_ms":  latency_a,
                "memories":    [],
            },
            "test_b": {
                "response":    response_b,
                "latency_ms":  latency_b,
                "memories":    [m["content"] for m in retrieved],
                "top_memory_scores": [round(m.get("retrieval_score", 0), 4) for m in retrieved],
            },
            "rating_a":  None,   # filled in by human rater below
            "rating_b":  None,
        }
        results.append(result)
        print(f"         A: {latency_a}ms | B: {latency_b}ms | "
              f"Memories injected: {len(retrieved)}\n")

    # ── Human rating loop ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RATING SESSION — Compare A vs B for each prompt")
    print("Rate each response 1-5 (1=poor, 5=excellent)")
    print("="*60)

    for r in results:
        print(f"\n{'─'*60}")
        print(f"Prompt {r['prompt_id']}: {r['prompt']}")
        print(f"\n── TEST A (no memory) ──")
        print(textwrap.fill(r["test_a"]["response"][:500], width=70))
        print(f"\n── TEST B (with {len(r['test_b']['memories'])} memories) ──")
        print(f"  Injected: {r['test_b']['memories'][:2]}")
        print(textwrap.fill(r["test_b"]["response"][:500], width=70))

        while True:
            try:
                ra = input(f"\nRate A (1-5): ").strip()
                rb = input(f"Rate B (1-5): ").strip()
                r["rating_a"] = int(ra)
                r["rating_b"] = int(rb)
                break
            except (ValueError, EOFError):
                print("Enter a number 1-5. Ctrl+C to skip remaining.")
                break

    # ── Summary ─────────────────────────────────────────────────────────────
    rated = [r for r in results if r["rating_a"] and r["rating_b"]]
    b_wins = sum(1 for r in rated if r["rating_b"] > r["rating_a"])
    a_wins = sum(1 for r in rated if r["rating_a"] > r["rating_b"])
    ties   = sum(1 for r in rated if r["rating_a"] == r["rating_b"])

    avg_a  = sum(r["rating_a"] for r in rated) / len(rated) if rated else 0
    avg_b  = sum(r["rating_b"] for r in rated) / len(rated) if rated else 0

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Prompts rated:     {len(rated)}/{len(results)}")
    print(f"Test B wins:       {b_wins}  (memory injection helped)")
    print(f"Test A wins:       {a_wins}  (no memory was better)")
    print(f"Ties:              {ties}")
    print(f"Average rating A:  {avg_a:.2f}/5")
    print(f"Average rating B:  {avg_b:.2f}/5")
    print(f"Improvement:       {avg_b - avg_a:+.2f} points")

    decision = "PASS — proceed to Experiment 2" if b_wins >= 7 else \
               "PARTIAL — review injection format before proceeding" if b_wins >= 5 else \
               "FAIL — memory injection not improving responses; revisit premise"
    print(f"\nDecision: {decision}")

    # ── Save results ─────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":  "exp_01_memory_injection",
        "model":       model,
        "run_at":      ts,
        "seed_facts":  SEED_FACTS,
        "b_wins":      b_wins,
        "a_wins":      a_wins,
        "ties":        ties,
        "avg_rating_a": avg_a,
        "avg_rating_b": avg_b,
        "decision":    decision,
        "prompts":     results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))

    txt_path = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 1 — Memory Injection\n"
        f"Run: {ts} | Model: {model}\n"
        f"B wins: {b_wins}/10 | Avg A: {avg_a:.2f} | Avg B: {avg_b:.2f}\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
