"""
Experiment 2 — Which Injection Format Works Best?
==================================================
Same memories, same prompts, three formats:
  A — JSON object
  B — Prose paragraph
  C — Structured [MEMORY: ...] tags

Determines the injection format used in working_memory.py.
Run: python3 POC/experiments/experiment_02/run.py
"""

import sys, json, time, pathlib, datetime, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from core.retrieval             import retrieve_tfidf
from core.working_memory        import build_prompt

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_FACTS = [
    {"content": "user prefers direct responses without preamble or filler phrases",
     "category": "preference", "confidence": 0.9},
    {"content": "user is building Project Jarvis in Python 3.12 — local-first AI memory layer",
     "category": "goal", "confidence": 0.95},
    {"content": "user has 10 years of software engineering experience; expert-level Python",
     "category": "expertise", "confidence": 0.9},
    {"content": "user dislikes over-explained answers; wants concise, direct answers",
     "category": "preference", "confidence": 0.9},
    {"content": "user is comfortable with SQLite, networkx, and the Ollama local API",
     "category": "expertise", "confidence": 0.85},
]

TEST_PROMPTS = [
    "How should I structure my SQLite schema for storing memory embeddings?",
    "What's the best approach to TF-IDF retrieval without scikit-learn?",
    "Should I use async or sync HTTP calls to Ollama in the PoC?",
    "How do I handle JSON parse failures from LLM extraction prompts?",
    "What's the simplest working memory builder I can write today?",
]

FORMATS = ["json", "prose", "structured"]

FORMAT_DESCRIPTIONS = {
    "json":       'JSON: {"user_context": [{"fact": "...", "confidence": 0.9}]}',
    "prose":      "Prose: 'Context about this user: They prefer direct responses...'",
    "structured": "[MEMORY: user prefers direct responses | confidence: high]",
}


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 2 — Injection Format Comparison")
    print(f"Model: {model} | Formats: {FORMATS}")
    print(f"{'='*60}\n")

    conn = fresh_db()
    seed_memories(conn, SEED_FACTS)
    memories = get_memories(conn)

    all_results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        retrieved = retrieve_tfidf(prompt, memories, top_n=4)
        print(f"\nPrompt {i}/{len(TEST_PROMPTS)}: {prompt[:60]}...")
        prompt_results = {"prompt_id": i, "prompt": prompt, "formats": {}}

        for fmt in FORMATS:
            full_prompt = build_prompt(prompt, retrieved, fmt=fmt)
            response, latency_ms = generate(full_prompt, model=model)
            prompt_results["formats"][fmt] = {
                "response":   response,
                "latency_ms": latency_ms,
                "prompt_len": len(full_prompt),
                "rating":     None,
            }
            print(f"  [{fmt:10s}] {latency_ms}ms | prompt_len={len(full_prompt)}")

        all_results.append(prompt_results)

    # ── Rating loop ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RATING — For each prompt, rate all 3 formats (1-5)")
    print("Focus: Does the response ACTUALLY USE the injected context?")
    print("="*60)

    for r in all_results:
        print(f"\n{'─'*60}")
        print(f"Prompt {r['prompt_id']}: {r['prompt']}")
        print(f"Memories injected: {[m['content'][:50] for m in retrieved][:2]}")

        for fmt in FORMATS:
            data = r["formats"][fmt]
            print(f"\n── Format {fmt.upper()} ──")
            print(f"  Context header: {FORMAT_DESCRIPTIONS[fmt]}")
            print(textwrap.fill(data["response"][:500], width=70))

        for fmt in FORMATS:
            while True:
                try:
                    rating = input(f"\nRate {fmt} (1-5): ").strip()
                    r["formats"][fmt]["rating"] = int(rating)
                    break
                except (ValueError, EOFError):
                    break

    # ── Summary ──────────────────────────────────────────────────────────────
    totals = {fmt: [] for fmt in FORMATS}
    for r in all_results:
        for fmt in FORMATS:
            rating = r["formats"][fmt].get("rating")
            if rating:
                totals[fmt].append(rating)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    avg_scores = {}
    for fmt in FORMATS:
        if totals[fmt]:
            avg = sum(totals[fmt]) / len(totals[fmt])
            avg_scores[fmt] = avg
            print(f"  {fmt:12s}: avg {avg:.2f}/5 ({len(totals[fmt])} ratings)")

    if avg_scores:
        winner = max(avg_scores, key=avg_scores.get)
        print(f"\nRecommended format: {winner.upper()}")
        print(f"Set INJECTION_FORMAT = '{winner}' in POC/core/jarvis_cli.py")
    else:
        winner = "structured"
        print("No ratings collected. Default recommendation: structured")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":     "exp_02_injection_format",
        "model":          model,
        "run_at":         ts,
        "avg_scores":     avg_scores,
        "recommended":    winner,
        "prompts":        all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 2 — Injection Format\n"
        f"Run: {ts} | Model: {model}\n"
        f"Avg scores: { {k: round(v,2) for k,v in avg_scores.items()} }\n"
        f"Recommended format: {winner}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
