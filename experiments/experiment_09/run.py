"""
Experiment 9 — Disinhibition Routing vs Flat TF-IDF End-to-End
===============================================================
Question: Does routing through retrieve_disinhibition() beat flat
retrieve_tfidf() end-to-end?

Method:
- 20 seed memories covering all 6 task types
- 12 test prompts (2 per task type) with known correct labels
- Method A: retrieve_tfidf → build_prompt → generate
- Method B: classify task type → retrieve_disinhibition → build_prompt → generate
- LLM judge compares A vs B per prompt

Pass criterion: b_wins >= 8 AND correct_classifications >= 10

Run: python3 POC/experiments/experiment_09/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama  import generate, pick_model
from experiments.shared.db      import fresh_db, seed_memories, get_memories
from experiments.shared.judge   import compare
from core.retrieval             import retrieve_tfidf, retrieve_disinhibition
from core.working_memory        import build_prompt

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Classification prompt (exact copy from spec) ──────────────────────────────

CLASSIFICATION_PROMPT = """Classify this request into exactly one of these categories:
PLANNING | RESEARCH | EXECUTION | EMOTIONAL | REFLECTION | LEARNING

Definitions:
- PLANNING: organising future work, priorities, schedules, task sequencing
  e.g. "Help me plan my week", "What should I prioritise?"
- RESEARCH: retrieving or synthesising information — recall from past reading,
  finding what exists in a field, or surveying the landscape of a topic.
  The user wants information gathered, not taught from scratch.
  e.g. "What do I know about X from my notes?", "What approaches exist for X?",
       "Summarise what I've read about X", "What's the current state of X?"
- EXECUTION: performing an action, writing code, creating something, sending something
  e.g. "Write the Python code for X", "Create a ticket for Y"
- EMOTIONAL: expressing or processing feelings, stress, frustration, joy
  e.g. "I'm feeling overwhelmed", "I'm frustrated that X isn't working"
- REFLECTION: looking back at past events, habits, patterns, lessons learned
  e.g. "What patterns have I noticed?", "Looking back at this week..."
- LEARNING: being taught a concept from scratch — explanation or tutorial request
  where the user wants to understand something new, not recall existing info.
  e.g. "Explain how X works", "Teach me about X", "How does X differ from Y?"

Key distinction — RESEARCH vs LEARNING:
  RESEARCH = "find / recall / what exists / what have I read"
  LEARNING = "explain to me / teach me / how does this work"

Request: {prompt}

Reply with ONLY the category name in capitals. Nothing else."""

CATEGORIES = ["PLANNING", "RESEARCH", "EXECUTION", "EMOTIONAL", "REFLECTION", "LEARNING"]

# ── 20 seed memories covering all 6 task types ────────────────────────────────

SEED_MEMORIES = [
    # preferences (4)
    {"content": "user prefers direct responses without preamble",     "category": "preference", "confidence": 0.9},
    {"content": "user dislikes over-explained answers",               "category": "preference", "confidence": 0.9},
    {"content": "user prefers concise code examples",                 "category": "preference", "confidence": 0.85},
    # goals (3)
    {"content": "user is building Project Jarvis — local AI memory layer in Python", "category": "goal", "confidence": 0.95},
    {"content": "user wants Jarvis to work offline",                  "category": "goal",       "confidence": 0.9},
    {"content": "user plans to build an MCP server for Claude Desktop","category": "goal",       "confidence": 0.85},
    # expertise (3)
    {"content": "user has 10 years of Python experience; expert level","category": "expertise",  "confidence": 0.9},
    {"content": "user knows TF-IDF, SQLite, networkx, and Ollama API", "category": "expertise",  "confidence": 0.85},
    {"content": "user is comfortable with machine learning basics",    "category": "expertise",  "confidence": 0.75},
    # patterns (3)
    {"content": "user works in 2-hour morning focus blocks",           "category": "pattern",    "confidence": 0.85},
    {"content": "user finds context switching disruptive",             "category": "pattern",    "confidence": 0.8},
    {"content": "user journals when stuck on hard problems",           "category": "pattern",    "confidence": 0.7},
    # general (8)
    {"content": "user is running Ollama locally on MacBook Pro M3 Pro with 36GB RAM", "category": "general", "confidence": 0.9},
    {"content": "user is building Jarvis as a solo project",           "category": "general",    "confidence": 0.9},
    {"content": "user tracks tasks in Linear",                         "category": "general",    "confidence": 0.8},
    {"content": "user uses GitHub for version control",                "category": "general",    "confidence": 0.8},
    {"content": "user prefers minimal Python dependencies",            "category": "general",    "confidence": 0.85},
    {"content": "user has tried Claude Projects and found it limited", "category": "general",    "confidence": 0.7},
    {"content": "user is in the Pacific timezone",                     "category": "general",    "confidence": 0.6},
    {"content": "user has worked at startups primarily",               "category": "general",    "confidence": 0.55},
]

# ── 12 test prompts (2 per category, with known labels) ──────────────────────

TEST_PROMPTS = [
    {"prompt": "Help me prioritise the next two weeks of Jarvis development.",                       "task_type": "PLANNING"},
    {"prompt": "What should I tackle first: the MCP server or the memory decay job?",               "task_type": "PLANNING"},
    {"prompt": "What do I know about local LLM memory implementations from my reading?",            "task_type": "RESEARCH"},
    {"prompt": "What approaches exist for confidence scoring in memory systems?",                   "task_type": "RESEARCH"},
    {"prompt": "Write the Python function that checks if a memory is a near-duplicate.",            "task_type": "EXECUTION"},
    {"prompt": "Create a Linear ticket for adding confidence decay to the memory store.",           "task_type": "EXECUTION"},
    {"prompt": "I'm frustrated — the extraction is still only hitting 57% recall.",                 "task_type": "EMOTIONAL"},
    {"prompt": "Feeling burnt out after two weeks of experiments. Need a break.",                   "task_type": "EMOTIONAL"},
    {"prompt": "Looking back at the 8 PoC experiments, what were the biggest surprises?",           "task_type": "REFLECTION"},
    {"prompt": "What patterns have I noticed in how I approach architecture decisions?",            "task_type": "REFLECTION"},
    {"prompt": "Explain how TF-IDF IDF weighting works when the corpus is small.",                  "task_type": "LEARNING"},
    {"prompt": "How does the VIP-cell disinhibition mechanism work in neuroscience?",               "task_type": "LEARNING"},
]


def classify_prompt(prompt: str, model: str) -> str:
    """Classify prompt into one of 6 task types. Returns uppercased category name."""
    full = CLASSIFICATION_PROMPT.format(prompt=prompt)
    raw, _ = generate(full, model=model, temperature=0.0, max_tokens=20)
    predicted = raw.strip().upper()
    matched = next((c for c in CATEGORIES if c in predicted), None)
    return matched if matched else "UNKNOWN"


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 9 — Disinhibition Routing vs Flat TF-IDF")
    print(f"Model: {model} | 20 memories | 12 test prompts")
    print(f"{'='*60}\n")

    conn = fresh_db()
    seed_memories(conn, SEED_MEMORIES)
    memories = get_memories(conn)
    print(f"Seeded {len(memories)} memories.\n")

    all_results = []
    b_wins = 0
    a_wins = 0
    ties   = 0
    correct_classifications = 0

    for i, test in enumerate(TEST_PROMPTS, 1):
        prompt        = test["prompt"]
        true_type     = test["task_type"]

        print(f"\n[{i:2d}/12] {true_type}: {prompt[:60]}...")

        # ── Classify (for Method B) ───────────────────────────────────────────
        classified_as = classify_prompt(prompt, model)
        is_correct    = (classified_as == true_type)
        if is_correct:
            correct_classifications += 1
        print(f"       Classified as: {classified_as} ({'correct' if is_correct else 'WRONG, expected ' + true_type})")

        # ── Method A: flat TF-IDF ─────────────────────────────────────────────
        top_a    = retrieve_tfidf(prompt, memories, top_n=4, confidence_weight=True)
        prompt_a = build_prompt(prompt, top_a, fmt="structured")
        resp_a, lat_a = generate(prompt_a, model=model, max_tokens=300)

        # ── Method B: disinhibition routing ───────────────────────────────────
        task_for_b = classified_as if classified_as in CATEGORIES else true_type
        top_b      = retrieve_disinhibition(prompt, task_for_b, memories, top_n=4, confidence_weight=True)
        prompt_b   = build_prompt(prompt, top_b, fmt="structured")
        resp_b, lat_b = generate(prompt_b, model=model, max_tokens=300)

        # ── LLM judge ────────────────────────────────────────────────────────
        top_mem_contents = [m["content"] for m in top_b[:4]]
        judge_a, judge_b = compare(prompt, resp_a, resp_b,
                                   memories=top_mem_contents, model=model)

        if judge_b > judge_a:
            winner = "B"
            b_wins += 1
        elif judge_a > judge_b:
            winner = "A"
            a_wins += 1
        else:
            winner = "TIE"
            ties += 1

        print(f"       Judge scores — A:{judge_a}  B:{judge_b}  winner:{winner}  "
              f"({lat_a}ms / {lat_b}ms)")

        all_results.append({
            "prompt_id":       i,
            "prompt":          prompt,
            "task_type":       true_type,
            "classified_as":   classified_as,
            "classification_correct": is_correct,
            "method_a": {
                "top_memories":  [m["content"][:80] for m in top_a],
                "response":      resp_a,
                "latency_ms":    lat_a,
                "judge_score":   judge_a,
            },
            "method_b": {
                "top_memories":  [m["content"][:80] for m in top_b],
                "response":      resp_b,
                "latency_ms":    lat_b,
                "judge_score":   judge_b,
            },
            "winner": winner,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Method B wins:          {b_wins}/12")
    print(f"Method A wins:          {a_wins}/12")
    print(f"Ties:                   {ties}/12")
    print(f"Correct classifications: {correct_classifications}/12")

    # Decision
    if b_wins >= 8 and correct_classifications >= 10:
        decision = "PASS — use disinhibition routing in jarvis_cli.py"
    elif b_wins >= 6 or correct_classifications >= 8:
        decision = "PARTIAL — disinhibition helps but classification needs work"
    else:
        decision = "FAIL — flat TF-IDF performs equally or better"

    print(f"\nDecision: {decision}")

    # ── Save results ─────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":              "exp_09_disinhibition_routing",
        "model":                   model,
        "run_at":                  ts,
        "b_wins":                  b_wins,
        "a_wins":                  a_wins,
        "ties":                    ties,
        "correct_classifications": correct_classifications,
        "pass_criterion":          "b_wins >= 8 AND correct_classifications >= 10",
        "decision":                decision,
        "results":                 all_results,
    }

    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))

    txt_path = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 9 — Disinhibition Routing vs Flat TF-IDF\n"
        f"Run: {ts} | Model: {model}\n"
        f"B wins: {b_wins}/12 | A wins: {a_wins}/12 | Ties: {ties}/12\n"
        f"Correct classifications: {correct_classifications}/12\n"
        f"Decision: {decision}\n"
    )

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
