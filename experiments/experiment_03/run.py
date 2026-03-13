"""
Experiment 3 — How Many Memories Before Quality Drops?
=======================================================
Tests N = 2, 4, 8, 16, 32 memories injected per prompt.
Measures: latency, response quality rating, memory utilisation.
Sets: top_N parameter in jarvis_cli.py

Run: python3 POC/experiments/experiment_03/run.py
"""

import sys, json, time, pathlib, datetime, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from core.retrieval             import retrieve_tfidf
from core.working_memory        import build_prompt

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 32 memories to test with — mix of relevant and irrelevant
ALL_MEMORIES = [
    # Highly relevant
    {"content": "user prefers direct responses without preamble",
     "category": "preference", "confidence": 0.9},
    {"content": "user is building Project Jarvis in Python 3.12",
     "category": "goal", "confidence": 0.95},
    {"content": "user has 10 years of Python experience; expert level",
     "category": "expertise", "confidence": 0.9},
    {"content": "user dislikes over-explained answers",
     "category": "preference", "confidence": 0.9},
    {"content": "user prefers concise code examples over pseudocode",
     "category": "preference", "confidence": 0.85},
    # Moderately relevant
    {"content": "user uses SQLite for the episodic log in Jarvis",
     "category": "general", "confidence": 0.8},
    {"content": "user is testing with llama3:latest on Ollama",
     "category": "general", "confidence": 0.85},
    {"content": "user is interested in memory retrieval algorithms",
     "category": "expertise", "confidence": 0.75},
    # Weakly relevant
    {"content": "user likes hiking on weekends",
     "category": "pattern", "confidence": 0.6},
    {"content": "user prefers morning work sessions",
     "category": "pattern", "confidence": 0.7},
    {"content": "user reads technical papers regularly",
     "category": "pattern", "confidence": 0.7},
    {"content": "user has a MacBook Pro M3 Pro with 36GB RAM",
     "category": "general", "confidence": 0.9},
    # Mostly irrelevant padding
    {"content": "user mentioned they like jazz music",
     "category": "pattern", "confidence": 0.5},
    {"content": "user has a dog named Max",
     "category": "general", "confidence": 0.55},
    {"content": "user's favourite programming language is Python",
     "category": "expertise", "confidence": 0.85},
    {"content": "user is in the Pacific timezone",
     "category": "general", "confidence": 0.6},
    {"content": "user prefers VS Code over other editors",
     "category": "preference", "confidence": 0.7},
    {"content": "user mentioned growing up in London",
     "category": "general", "confidence": 0.45},
    {"content": "user drinks coffee, not tea",
     "category": "pattern", "confidence": 0.5},
    {"content": "user has a standing desk setup",
     "category": "general", "confidence": 0.55},
    {"content": "user likes to use type hints in Python",
     "category": "preference", "confidence": 0.75},
    {"content": "user mentioned watching sci-fi films",
     "category": "pattern", "confidence": 0.4},
    {"content": "user has tried several AI tools before Jarvis",
     "category": "general", "confidence": 0.65},
    {"content": "user finds context switching disruptive",
     "category": "pattern", "confidence": 0.7},
    {"content": "user uses GitHub for version control",
     "category": "general", "confidence": 0.8},
    {"content": "user mentioned reading Clean Code by Robert Martin",
     "category": "expertise", "confidence": 0.6},
    {"content": "user prefers writing tests after implementation in PoC phase",
     "category": "preference", "confidence": 0.65},
    {"content": "user's secondary language is JavaScript",
     "category": "expertise", "confidence": 0.5},
    {"content": "user mentioned they meditate in the mornings",
     "category": "pattern", "confidence": 0.45},
    {"content": "user has worked at startups primarily",
     "category": "general", "confidence": 0.55},
    {"content": "user is interested in building a SaaS product eventually",
     "category": "goal", "confidence": 0.6},
    {"content": "user prefers documentation that explains the 'why' not just the 'how'",
     "category": "preference", "confidence": 0.7},
]

TEST_PROMPTS = [
    "How should I structure my Python module for memory retrieval?",
    "What's the best way to handle schema migrations in SQLite?",
    "Give me a concise overview of TF-IDF scoring.",
    "How should I think about the context injection format?",
    "What's a good approach for session management in my CLI tool?",
]

N_VALUES = [2, 4, 8, 16, 32]


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print(f"Experiment 3 — Memory Count Scaling")
    print(f"Model: {model} | Testing N = {N_VALUES}")
    print(f"{'='*60}\n")

    conn = fresh_db()
    seed_memories(conn, ALL_MEMORIES)
    memories = get_memories(conn)
    print(f"Seeded {len(memories)} memories.\n")

    all_results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\nPrompt {i}/{len(TEST_PROMPTS)}: {prompt[:60]}")
        prompt_result = {"prompt_id": i, "prompt": prompt, "by_n": {}}

        for n in N_VALUES:
            retrieved  = retrieve_tfidf(prompt, memories, top_n=n)
            full_prompt = build_prompt(prompt, retrieved, fmt="structured")
            prompt_len  = len(full_prompt)

            response, latency_ms = generate(full_prompt, model=model, max_tokens=512)

            prompt_result["by_n"][n] = {
                "n_injected":  len(retrieved),
                "prompt_len":  prompt_len,
                "latency_ms":  latency_ms,
                "response":    response,
                "rating":      None,
            }
            print(f"  N={n:2d}: {latency_ms:5d}ms | prompt_len={prompt_len:5d} chars | "
                  f"response_len={len(response):4d} chars")

        all_results.append(prompt_result)

    # ── Automated checks ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LATENCY ANALYSIS (automated)")
    print(f"{'='*60}")
    for n in N_VALUES:
        latencies = [
            r["by_n"][n]["latency_ms"]
            for r in all_results
            if n in r["by_n"]
        ]
        if latencies:
            avg_l = sum(latencies) / len(latencies)
            print(f"  N={n:2d}: avg latency = {avg_l:.0f}ms")

    # ── Rating loop ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RATING — Compare quality at each N value (1-5)")
    print("Focus: coherence, does it use memories, any hallucinations?")
    print(f"{'='*60}")

    for r in all_results:
        print(f"\n{'─'*60}")
        print(f"Prompt {r['prompt_id']}: {r['prompt']}")
        for n in N_VALUES:
            data = r["by_n"][n]
            print(f"\n── N={n} ({data['n_injected']} memories, {data['latency_ms']}ms) ──")
            print(textwrap.fill(data["response"][:400], width=70))
            while True:
                try:
                    rating = input(f"Rate N={n} (1-5, or Enter to skip): ").strip()
                    if rating:
                        r["by_n"][n]["rating"] = int(rating)
                    break
                except (ValueError, EOFError):
                    break

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("QUALITY SUMMARY BY N")
    print(f"{'='*60}")
    best_n = None
    best_avg = 0
    for n in N_VALUES:
        ratings = [r["by_n"][n]["rating"] for r in all_results if r["by_n"][n].get("rating")]
        if ratings:
            avg = sum(ratings) / len(ratings)
            print(f"  N={n:2d}: avg quality = {avg:.2f}/5")
            if avg > best_avg:
                best_avg = avg
                best_n = n

    if best_n:
        print(f"\nRecommended top_N: {best_n}")
        print(f"Set TOP_N = {best_n} in POC/core/jarvis_cli.py")
    else:
        print("No ratings collected. Recommended default: top_N = 4")
        best_n = 4

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":  "exp_03_memory_count",
        "model":       model,
        "run_at":      ts,
        "n_values":    N_VALUES,
        "recommended_top_n": best_n,
        "prompts":     all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 3 — Memory Count\n"
        f"Run: {ts} | Model: {model}\n"
        f"Recommended top_N: {best_n}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
