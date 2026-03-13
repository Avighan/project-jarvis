"""
Experiment 8 — Confidence Decay: Does It Help or Add Noise?
============================================================
100 memories with varying confidence levels.
Test A: retrieve top-N by TF-IDF relevance score only.
Test B: retrieve top-N by (relevance × confidence).

Decision: if Test B improves response quality on ≥3/5 prompts → activate
          confidence weighting in jarvis_cli.py.

Run: python3 POC/experiments/experiment_08/run.py
"""

import sys, json, pathlib, datetime, textwrap, random
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from core.retrieval             import retrieve_tfidf
from core.working_memory        import build_prompt

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 100 memories: mix of high, medium, low confidence ────────────────────────
def make_memories() -> list[dict]:
    """Generate 100 memories spanning all confidence tiers."""
    # High confidence (0.8-1.0): recent, repeatedly confirmed facts
    high = [
        {"content": "user prefers direct answers without preamble", "category": "preference", "confidence": 0.95},
        {"content": "user is building Project Jarvis in Python", "category": "goal", "confidence": 0.95},
        {"content": "user has 10 years of Python experience", "category": "expertise", "confidence": 0.90},
        {"content": "user uses MacBook Pro M3 Pro with 36GB RAM", "category": "general", "confidence": 0.95},
        {"content": "user wants Jarvis to work completely offline", "category": "goal", "confidence": 0.90},
        {"content": "user has Ollama running with llama3:latest", "category": "general", "confidence": 0.90},
        {"content": "user dislikes over-explained responses", "category": "preference", "confidence": 0.90},
        {"content": "user prefers minimal Python dependencies", "category": "preference", "confidence": 0.85},
        {"content": "user is testing PoC with SQLite and networkx", "category": "general", "confidence": 0.85},
        {"content": "user wants MCP server accessible from Claude Desktop", "category": "goal", "confidence": 0.88},
        {"content": "user works in 2-hour morning focus blocks", "category": "pattern", "confidence": 0.85},
        {"content": "user has 10+ installed Python packages in this project", "category": "general", "confidence": 0.80},
        {"content": "user prefers synchronous Python code for PoC", "category": "preference", "confidence": 0.82},
        {"content": "user tracks tasks in Linear", "category": "pattern", "confidence": 0.80},
        {"content": "user prefers Python type hints in all function signatures", "category": "preference", "confidence": 0.83},
    ]

    # Medium confidence (0.45-0.75): older or partially corroborated
    medium = [
        {"content": "user mentioned wanting to add voice input eventually", "category": "goal", "confidence": 0.65},
        {"content": "user may prefer VS Code over vim for editing", "category": "preference", "confidence": 0.55},
        {"content": "user reads technical papers in the evening", "category": "pattern", "confidence": 0.70},
        {"content": "user seems interested in Rust but hasn't used it recently", "category": "expertise", "confidence": 0.50},
        {"content": "user mentioned being in a startup environment", "category": "general", "confidence": 0.60},
        {"content": "user has used Flask before for web APIs", "category": "expertise", "confidence": 0.70},
        {"content": "user probably prefers dark mode in their editor", "category": "preference", "confidence": 0.55},
        {"content": "user seems to work better with music in the background", "category": "pattern", "confidence": 0.50},
        {"content": "user may want to open source Jarvis eventually", "category": "goal", "confidence": 0.60},
        {"content": "user mentioned using GitHub Actions once for CI", "category": "general", "confidence": 0.65},
        {"content": "user has some familiarity with Docker", "category": "expertise", "confidence": 0.60},
        {"content": "user sometimes journals when stuck on problems", "category": "pattern", "confidence": 0.65},
        {"content": "user mentioned reading Clean Code several years ago", "category": "expertise", "confidence": 0.55},
        {"content": "user seems to prefer single-file scripts for experiments", "category": "preference", "confidence": 0.65},
        {"content": "user might have used pandas in a previous job", "category": "expertise", "confidence": 0.50},
    ]

    # Low confidence (0.1-0.4): contradicted, stale, or weak signal
    low = [
        {"content": "user prefers verbose detailed explanations (contradicted)", "category": "preference", "confidence": 0.15},
        {"content": "user mentioned they might switch to TypeScript", "category": "preference", "confidence": 0.25},
        {"content": "user once said they use Notion for tasks (superseded by Linear)", "category": "pattern", "confidence": 0.20},
        {"content": "user's dog is named Max (mentioned once, unverified)", "category": "general", "confidence": 0.30},
        {"content": "user might prefer async Python (contradicted by sync preference)", "category": "preference", "confidence": 0.20},
        {"content": "user once mentioned wanting to learn Go", "category": "expertise", "confidence": 0.30},
        {"content": "user may prefer working late at night (contradicted by morning preference)", "category": "pattern", "confidence": 0.15},
        {"content": "user mentioned liking jazz music once", "category": "pattern", "confidence": 0.35},
        {"content": "user might have a home office setup (uncertain)", "category": "general", "confidence": 0.35},
        {"content": "user possibly prefers Mac over Linux (mentioned in passing)", "category": "general", "confidence": 0.40},
        {"content": "user may have tried Cursor IDE briefly", "category": "general", "confidence": 0.35},
        {"content": "user once asked about React — may be exploring frontend", "category": "expertise", "confidence": 0.30},
        {"content": "user might drink tea sometimes (contradicts coffee mention)", "category": "pattern", "confidence": 0.20},
        {"content": "user mentioned wanting to travel more in passing", "category": "pattern", "confidence": 0.25},
        {"content": "user once said they find OOP overused (single mention)", "category": "preference", "confidence": 0.35},
    ]

    # Padding to reach 100 (neutral, medium confidence, unrelated)
    padding = []
    padding_templates = [
        ("user has not mentioned family background", "general", 0.5),
        ("user uses Safari as primary browser", "general", 0.55),
        ("user is interested in AI safety topics", "expertise", 0.60),
        ("user has read about LangChain but not used it", "expertise", 0.55),
        ("user mentioned the weather once in conversation", "general", 0.30),
        ("user has a standing desk (mentioned once)", "general", 0.45),
        ("user tracks Hacker News regularly", "pattern", 0.65),
        ("user has contributed to open source in the past", "general", 0.60),
        ("user has tried Claude Projects feature", "general", 0.70),
        ("user mentioned energy dips after lunch", "pattern", 0.55),
        ("user is interested in privacy-preserving AI", "preference", 0.75),
        ("user has experience with SQL databases generally", "expertise", 0.70),
        ("user mentioned reading about RLHF", "expertise", 0.55),
        ("user uses a password manager", "general", 0.50),
        ("user follows several AI researchers on Twitter", "pattern", 0.60),
        ("user is aware of the context window limitations of LLMs", "expertise", 0.75),
        ("user mentioned they benchmark their code", "preference", 0.60),
        ("user has worked on distributed systems before", "expertise", 0.55),
        ("user prefers functional programming style in places", "preference", 0.60),
        ("user mentioned wanting to write a technical blog post", "goal", 0.55),
        ("user is aware of RAG architectures", "expertise", 0.70),
        ("user has used Redis for caching in a previous role", "expertise", 0.50),
        ("user prefers writing tests for public APIs", "preference", 0.65),
        ("user mentioned using a VPN", "general", 0.45),
        ("user is interested in personal knowledge management", "goal", 0.70),
        ("user has tried Notion AI and found it limited", "general", 0.65),
        ("user has some familiarity with vector databases", "expertise", 0.60),
        ("user mentioned listening to podcasts during walks", "pattern", 0.50),
        ("user has strong opinions on code formatting standards", "preference", 0.65),
        ("user is interested in the Anthropic model card research", "expertise", 0.60),
        ("user mentioned their work involves both frontend and backend", "general", 0.55),
        ("user prefers environment variables for secrets management", "preference", 0.70),
        ("user has experience with CI/CD pipelines", "expertise", 0.60),
        ("user tracks version history with git carefully", "preference", 0.75),
        ("user mentioned context switching between projects is frustrating", "pattern", 0.70),
        ("user has used Jupyter notebooks for data exploration", "expertise", 0.55),
        ("user is interested in the intersection of AI and productivity", "goal", 0.80),
        ("user mentioned they prefer async code in web servers", "preference", 0.60),
        ("user has used FastAPI for building APIs", "expertise", 0.65),
        ("user mentioned having a good handle on system design", "expertise", 0.65),
        ("user occasionally watches programming tutorials on YouTube", "pattern", 0.50),
        ("user mentioned reading The Pragmatic Programmer", "expertise", 0.55),
        ("user prefers explicit error messages in code", "preference", 0.70),
        ("user is aware of the CAP theorem", "expertise", 0.65),
        ("user has familiarity with shell scripting (bash/zsh)", "expertise", 0.70),
        ("user prefers self-documenting code over comments", "preference", 0.65),
        ("user mentioned interest in ergonomics and keyboard health", "pattern", 0.45),
        ("user is comfortable reading academic papers", "expertise", 0.65),
        ("user has mentioned using time-blocking techniques", "pattern", 0.60),
        ("user is thoughtful about technical debt decisions", "preference", 0.70),
        ("user mentioned they are deliberate about tool choices", "preference", 0.70),
        ("user has used Anthropic's Claude API directly before Jarvis", "general", 0.65),
        ("user thinks carefully before adding new dependencies", "preference", 0.75),
        ("user is building Jarvis as a solo project", "general", 0.90),
        ("user mentioned they find async debugging harder than sync", "preference", 0.65),
        ("user has experience writing technical documentation", "expertise", 0.60),
        ("user tracks metrics on their projects where possible", "preference", 0.65),
        ("user is interested in product-led growth for Jarvis", "goal", 0.65),
        ("user prefers monorepo structure for multi-component projects", "preference", 0.55),
        ("user has tried building chatbots before", "expertise", 0.55),
        ("user is interested in the alignment problem in AI", "expertise", 0.60),
        ("user mentioned being curious about Go's concurrency model", "expertise", 0.45),
        ("user prefers explicit over implicit in API design", "preference", 0.70),
        ("user has used UV for Python package management recently", "general", 0.65),
        ("user mentioned preferring local tooling over cloud tooling", "preference", 0.80),
        ("user is tracking the LLM space closely", "pattern", 0.80),
        ("user mentioned being excited about multi-modal AI capabilities", "expertise", 0.55),
        ("user has used Makefiles for project automation", "general", 0.50),
        ("user prefers writing Python scripts over Jupyter for production work", "preference", 0.70),
    ]
    for text, cat, conf in padding_templates[:70]:
        padding.append({"content": text, "category": cat, "confidence": conf})

    all_mems = high + medium + low + padding
    return all_mems[:100]


# 5 test prompts where a relevant memory exists at both high and low confidence
TEST_CASES = [
    {
        "prompt": "How should Jarvis format responses for this user?",
        "high_conf_answer": "user prefers direct answers without preamble",
        "low_conf_answer":  "user prefers verbose detailed explanations (contradicted)",
        "expected_winner": "high",
    },
    {
        "prompt": "What coding style does this user prefer for Python?",
        "high_conf_answer": "user prefers minimal Python dependencies",
        "low_conf_answer":  "user might prefer async Python (contradicted by sync preference)",
        "expected_winner": "high",
    },
    {
        "prompt": "When should the proactive daemon avoid sending notifications?",
        "high_conf_answer": "user works in 2-hour morning focus blocks",
        "low_conf_answer":  "user may prefer working late at night (contradicted by morning preference)",
        "expected_winner": "high",
    },
    {
        "prompt": "What task management tool does this user use?",
        "high_conf_answer": "user tracks tasks in Linear",
        "low_conf_answer":  "user once said they use Notion for tasks (superseded by Linear)",
        "expected_winner": "high",
    },
    {
        "prompt": "Should Jarvis explain things concisely or in detail?",
        "high_conf_answer": "user dislikes over-explained responses",
        "low_conf_answer":  "user prefers verbose detailed explanations (contradicted)",
        "expected_winner": "high",
    },
]


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 8 — Confidence Decay: Does It Help?")
    print(f"Model: {model} | 100 memories | 5 test prompts")
    print(f"{'='*60}\n")

    conn = fresh_db()
    memories_def = make_memories()
    seed_memories(conn, memories_def)
    memories = get_memories(conn)
    print(f"Seeded {len(memories)} memories.")
    high_conf = sum(1 for m in memories if m["confidence"] >= 0.8)
    med_conf  = sum(1 for m in memories if 0.45 <= m["confidence"] < 0.8)
    low_conf  = sum(1 for m in memories if m["confidence"] < 0.45)
    print(f"  High (≥0.8):      {high_conf}")
    print(f"  Medium (0.45-0.8): {med_conf}")
    print(f"  Low (<0.45):       {low_conf}\n")

    all_results = []
    b_better = 0

    for i, test in enumerate(TEST_CASES, 1):
        prompt = test["prompt"]
        print(f"\nTest {i}/5: {prompt}")

        # Method A — relevance only
        top_a = retrieve_tfidf(prompt, memories, top_n=4, confidence_weight=False)
        # Method B — relevance × confidence
        top_b = retrieve_tfidf(prompt, memories, top_n=4, confidence_weight=True)

        # Check if high-confidence answer is in top results
        def contains(results, text_fragment):
            return any(text_fragment[:40].lower() in m["content"].lower() for m in results)

        high_in_a = contains(top_a, test["high_conf_answer"])
        low_in_a  = contains(top_a, test["low_conf_answer"])
        high_in_b = contains(top_b, test["high_conf_answer"])
        low_in_b  = contains(top_b, test["low_conf_answer"])

        # Generate responses
        response_a, lat_a = generate(build_prompt(prompt, top_a, fmt="structured"), model=model, max_tokens=300)
        response_b, lat_b = generate(build_prompt(prompt, top_b, fmt="structured"), model=model, max_tokens=300)

        print(f"  Method A (relevance only):     high={high_in_a} low={low_in_a} ({lat_a}ms)")
        print(f"  Method B (relevance×confidence): high={high_in_b} low={low_in_b} ({lat_b}ms)")

        result = {
            "test_id":         i,
            "prompt":          prompt,
            "high_conf_target": test["high_conf_answer"],
            "low_conf_target":  test["low_conf_answer"],
            "method_a": {
                "has_high_conf": high_in_a,
                "has_low_conf":  low_in_a,
                "response":      response_a,
                "latency_ms":    lat_a,
                "top_memories":  [m["content"][:60] for m in top_a],
                "rating":        None,
            },
            "method_b": {
                "has_high_conf": high_in_b,
                "has_low_conf":  low_in_b,
                "response":      response_b,
                "latency_ms":    lat_b,
                "top_memories":  [m["content"][:60] for m in top_b],
                "rating":        None,
            },
        }
        all_results.append(result)

    # ── Rating loop ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RATING — Compare Method A vs B (1-5)")
    print("Focus: Does B surface more accurate/trustworthy memories?")
    print(f"{'='*60}")

    for r in all_results:
        print(f"\n{'─'*60}")
        print(f"Test {r['test_id']}: {r['prompt']}")
        print(f"\n── Method A (relevance only) ──")
        print(f"  Top memories: {r['method_a']['top_memories'][:2]}")
        print(textwrap.fill(r["method_a"]["response"][:400], width=70))
        print(f"\n── Method B (confidence-weighted) ──")
        print(f"  Top memories: {r['method_b']['top_memories'][:2]}")
        print(textwrap.fill(r["method_b"]["response"][:400], width=70))

        for method in ["method_a", "method_b"]:
            while True:
                try:
                    label = "A" if method == "method_a" else "B"
                    rating = input(f"Rate {label} (1-5): ").strip()
                    if rating:
                        r[method]["rating"] = int(rating)
                    break
                except (ValueError, EOFError):
                    break

    # ── Summary ──────────────────────────────────────────────────────────────
    rated = [r for r in all_results if r["method_a"].get("rating") and r["method_b"].get("rating")]
    b_better = sum(1 for r in rated if r["method_b"]["rating"] > r["method_a"]["rating"])
    a_better = sum(1 for r in rated if r["method_a"]["rating"] > r["method_b"]["rating"])
    ties     = sum(1 for r in rated if r["method_a"]["rating"] == r["method_b"]["rating"])

    # Automatic score: B prevents low-confidence from surfacing
    auto_b_better = sum(
        1 for r in all_results
        if r["method_b"]["has_high_conf"] and not r["method_b"]["has_low_conf"]
        and (not r["method_a"]["has_high_conf"] or r["method_a"]["has_low_conf"])
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Method B rated better: {b_better}/{len(rated)} prompts")
    print(f"Method A rated better: {a_better}/{len(rated)} prompts")
    print(f"Ties:                  {ties}/{len(rated)}")
    print(f"Auto-detected B advantage (suppresses contradicted facts): {auto_b_better}/5")

    if b_better >= 3 or auto_b_better >= 3:
        decision = (f"ACTIVATE — confidence weighting improves retrieval. "
                    f"Set CONFIDENCE_WEIGHT = True in jarvis_cli.py.")
    else:
        decision = (f"DEFER — no clear benefit at 100 memories. "
                    f"Implement schema, leave CONFIDENCE_WEIGHT = False. "
                    f"Re-test at 500+ memories.")

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":     "exp_08_confidence_decay",
        "model":          model,
        "run_at":         ts,
        "b_rated_better": b_better,
        "a_rated_better": a_better,
        "auto_b_better":  auto_b_better,
        "decision":       decision,
        "results":        all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 8 — Confidence Decay\n"
        f"Run: {ts} | Model: {model}\n"
        f"B better: {b_better}/5 | Auto B advantage: {auto_b_better}/5\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
