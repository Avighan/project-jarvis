"""
Project Jarvis PoC — Auto-run all 8 experiments using LLM-as-Judge.

Experiments 01 & 02: re-rate already-generated JSON results.
Experiments 03 & 08: run fresh generation + LLM judge.
Experiments 04-07:   fully automated (no rating needed).

Usage:
    python3 POC/auto_run_all.py
"""

import sys, json, pathlib, subprocess, datetime, textwrap, re

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "POC"))

from experiments.shared.ollama  import generate, pick_model
from experiments.shared.db      import fresh_db, seed_memories, get_memories
from experiments.shared.judge   import compare, rate
from core.retrieval              import retrieve_tfidf
from core.working_memory         import build_prompt, build_prompt_no_memory

EXPS_DIR = ROOT / "POC" / "experiments"
MODEL = pick_model()

SEP  = "=" * 60
SEP2 = "-" * 60

print(f"\n{SEP}")
print(f"Project Jarvis PoC — Auto-Run All Experiments")
print(f"Model: {MODEL}")
print(f"LLM-as-Judge: ON")
print(f"{SEP}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def latest_json(exp_num: str) -> pathlib.Path | None:
    """Return the most recent results JSON for a given experiment number."""
    results_dir = EXPS_DIR / f"experiment_{exp_num}" / "results"
    files = sorted(results_dir.glob("run_*.json"), reverse=True)
    return files[0] if files else None


def run_subprocess(exp_num: str) -> str:
    """Run an experiment as a subprocess and return stdout."""
    script = EXPS_DIR / f"experiment_{exp_num}" / "run.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True,
        cwd=str(ROOT / "POC"),
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  [WARN] exp_{exp_num} stderr: {result.stderr[:200]}")
    return result.stdout


# ── Experiment 01 — Re-rate from saved JSON ───────────────────────────────────

print(f"\n{'='*60}")
print("EXPERIMENT 01 — Memory Injection: Does It Matter?")
print(f"{'='*60}")

json01 = latest_json("01")
if json01:
    data01 = json.loads(json01.read_text())
    results01 = data01["prompts"]
    print(f"Loaded {len(results01)} prompts from {json01.name}")
    print("Running LLM judge on each A/B pair...\n")

    for r in results01:
        memories = r["test_b"]["memories"]
        ra, rb = compare(
            r["prompt"],
            r["test_a"]["response"],
            r["test_b"]["response"],
            memories=memories,
            model=MODEL,
        )
        r["rating_a"] = ra
        r["rating_b"] = rb
        winner = "B" if rb > ra else ("A" if ra > rb else "tie")
        print(f"  Prompt {r['prompt_id']:2d}: A={ra} B={rb} → {winner}  | {r['prompt'][:55]}...")

    rated = [r for r in results01 if r["rating_a"] and r["rating_b"]]
    b_wins = sum(1 for r in rated if r["rating_b"] > r["rating_a"])
    a_wins = sum(1 for r in rated if r["rating_a"] > r["rating_b"])
    ties   = sum(1 for r in rated if r["rating_a"] == r["rating_b"])
    avg_a  = sum(r["rating_a"] for r in rated) / len(rated)
    avg_b  = sum(r["rating_b"] for r in rated) / len(rated)

    decision01 = (
        "PASS — proceed to Experiment 2"           if b_wins >= 7 else
        "PARTIAL — review injection format"         if b_wins >= 5 else
        "FAIL — memory injection not improving"
    )

    print(f"\n  B wins: {b_wins}/10 | A wins: {a_wins}/10 | Ties: {ties}")
    print(f"  Avg A: {avg_a:.2f} | Avg B: {avg_b:.2f} | Δ: {avg_b-avg_a:+.2f}")
    print(f"  Decision: {decision01}")

    # Update JSON
    data01.update({
        "b_wins": b_wins, "a_wins": a_wins, "ties": ties,
        "avg_rating_a": avg_a, "avg_rating_b": avg_b,
        "decision": decision01,
    })
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPS_DIR / "experiment_01" / "results" / f"run_{ts}_auto.json"
    out_path.write_text(json.dumps(data01, indent=2))
    print(f"  Saved → {out_path.name}")
else:
    print("  [ERROR] No exp_01 results found. Run experiment_01/run.py first.")
    decision01 = "UNKNOWN"


# ── Experiment 02 — Re-rate from saved JSON ───────────────────────────────────

print(f"\n{'='*60}")
print("EXPERIMENT 02 — Injection Format (JSON vs Prose vs Structured)")
print(f"{'='*60}")

json02 = latest_json("02")
if json02:
    data02 = json.loads(json02.read_text())
    prompts02 = data02["prompts"]
    print(f"Loaded {len(prompts02)} prompts from {json02.name}")
    print("Running LLM judge on each format...\n")

    format_scores: dict[str, list[int]] = {"json": [], "prose": [], "structured": []}

    for r in prompts02:
        prompt = r["prompt"]
        memories_used = []  # exp02 retrieved per prompt but didn't save which — use hint
        for fmt in ["json", "prose", "structured"]:
            if fmt not in r.get("formats", {}):
                continue
            resp = r["formats"][fmt]["response"]
            score = rate(prompt, resp, model=MODEL)
            r["formats"][fmt]["rating"] = score
            format_scores[fmt].append(score)
        scores_str = "  ".join(f"{f}={r['formats'][f].get('rating','?')}" for f in ["json","prose","structured"] if f in r.get("formats",{}))
        print(f"  Prompt {r['prompt_id']}: {scores_str} | {prompt[:50]}...")

    avg_scores02 = {
        fmt: sum(v)/len(v) for fmt, v in format_scores.items() if v
    }
    winner02 = max(avg_scores02, key=avg_scores02.get) if avg_scores02 else "structured"

    print(f"\n  Avg scores: { {k: round(v,2) for k,v in avg_scores02.items()} }")
    print(f"  Recommended format: {winner02.upper()}")
    print(f"  → Set INJECTION_FORMAT = '{winner02}' in POC/core/jarvis_cli.py")

    data02.update({"avg_scores": avg_scores02, "recommended": winner02})
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPS_DIR / "experiment_02" / "results" / f"run_{ts}_auto.json"
    out_path.write_text(json.dumps(data02, indent=2))
    print(f"  Saved → {out_path.name}")
else:
    print("  [ERROR] No exp_02 results found.")
    winner02 = "structured"


# ── Experiment 03 — Run fresh + LLM judge ────────────────────────────────────

print(f"\n{'='*60}")
print("EXPERIMENT 03 — Memory Count Scaling (N = 2, 4, 8, 16, 32)")
print(f"{'='*60}")

ALL_MEMORIES_03 = [
    {"content": "user prefers direct responses without preamble",         "category": "preference", "confidence": 0.9},
    {"content": "user is building Project Jarvis in Python 3.12",        "category": "goal",       "confidence": 0.95},
    {"content": "user has 10 years of Python experience; expert level",  "category": "expertise",  "confidence": 0.9},
    {"content": "user dislikes over-explained answers",                  "category": "preference", "confidence": 0.9},
    {"content": "user prefers concise code examples over pseudocode",    "category": "preference", "confidence": 0.85},
    {"content": "user uses SQLite for the episodic log in Jarvis",       "category": "general",    "confidence": 0.8},
    {"content": "user is testing with llama3:latest on Ollama",          "category": "general",    "confidence": 0.85},
    {"content": "user is interested in memory retrieval algorithms",     "category": "expertise",  "confidence": 0.75},
    {"content": "user likes hiking on weekends",                         "category": "pattern",    "confidence": 0.6},
    {"content": "user prefers morning work sessions",                    "category": "pattern",    "confidence": 0.7},
    {"content": "user reads technical papers regularly",                 "category": "pattern",    "confidence": 0.7},
    {"content": "user has a MacBook Pro M3 Pro with 36GB RAM",           "category": "general",    "confidence": 0.9},
    {"content": "user mentioned they like jazz music",                   "category": "pattern",    "confidence": 0.5},
    {"content": "user has a dog named Max",                              "category": "general",    "confidence": 0.55},
    {"content": "user's favourite programming language is Python",       "category": "expertise",  "confidence": 0.85},
    {"content": "user is in the Pacific timezone",                       "category": "general",    "confidence": 0.6},
    {"content": "user prefers VS Code over other editors",               "category": "preference", "confidence": 0.7},
    {"content": "user mentioned growing up in London",                   "category": "general",    "confidence": 0.45},
    {"content": "user drinks coffee, not tea",                           "category": "pattern",    "confidence": 0.5},
    {"content": "user has a standing desk setup",                        "category": "general",    "confidence": 0.55},
    {"content": "user likes to use type hints in Python",                "category": "preference", "confidence": 0.75},
    {"content": "user mentioned watching sci-fi films",                  "category": "pattern",    "confidence": 0.4},
    {"content": "user has tried several AI tools before Jarvis",         "category": "general",    "confidence": 0.65},
    {"content": "user finds context switching disruptive",               "category": "pattern",    "confidence": 0.7},
    {"content": "user uses GitHub for version control",                  "category": "general",    "confidence": 0.8},
    {"content": "user mentioned reading Clean Code by Robert Martin",    "category": "expertise",  "confidence": 0.6},
    {"content": "user prefers writing tests after implementation in PoC","category": "preference", "confidence": 0.65},
    {"content": "user's secondary language is JavaScript",              "category": "expertise",  "confidence": 0.5},
    {"content": "user mentioned they meditate in the mornings",          "category": "pattern",    "confidence": 0.45},
    {"content": "user has worked at startups primarily",                 "category": "general",    "confidence": 0.55},
    {"content": "user is interested in building a SaaS product eventually","category": "goal",     "confidence": 0.6},
    {"content": "user prefers documentation that explains the 'why'",   "category": "preference", "confidence": 0.7},
]

TEST_PROMPTS_03 = [
    "How should I structure my Python module for memory retrieval?",
    "What's the best way to handle schema migrations in SQLite?",
    "Give me a concise overview of TF-IDF scoring.",
    "How should I think about the context injection format?",
    "What's a good approach for session management in my CLI tool?",
]
N_VALUES = [2, 4, 8, 16, 32]

conn03 = fresh_db()
seed_memories(conn03, ALL_MEMORIES_03)
mems03 = get_memories(conn03)
print(f"Seeded {len(mems03)} memories. Running {len(TEST_PROMPTS_03)} prompts × {len(N_VALUES)} N values...\n")

all_results03 = []
n_latencies: dict[int, list[int]] = {n: [] for n in N_VALUES}

for i, prompt in enumerate(TEST_PROMPTS_03, 1):
    pr = {"prompt_id": i, "prompt": prompt, "by_n": {}}
    print(f"  Prompt {i}/{len(TEST_PROMPTS_03)}: {prompt[:55]}...")
    for n in N_VALUES:
        retrieved  = retrieve_tfidf(prompt, mems03, top_n=n)
        full_p     = build_prompt(prompt, retrieved, fmt="structured")
        response, latency_ms = generate(full_p, model=MODEL, max_tokens=512)
        mem_list   = [m["content"] for m in retrieved]
        score      = rate(prompt, response, memories=mem_list[:3], model=MODEL)
        pr["by_n"][n] = {
            "n_injected":  len(retrieved),
            "prompt_len":  len(full_p),
            "latency_ms":  latency_ms,
            "response":    response,
            "rating":      score,
        }
        n_latencies[n].append(latency_ms)
        print(f"    N={n:2d}: {latency_ms:5d}ms | len={len(full_p):5d} | judge={score}")
    all_results03.append(pr)

best_n = None
best_avg = 0.0
print(f"\n  Quality by N:")
for n in N_VALUES:
    ratings = [r["by_n"][n]["rating"] for r in all_results03 if r["by_n"][n].get("rating")]
    avg_lat = sum(n_latencies[n])/len(n_latencies[n]) if n_latencies[n] else 0
    if ratings:
        avg_q = sum(ratings)/len(ratings)
        print(f"    N={n:2d}: quality={avg_q:.2f}/5  avg_latency={avg_lat:.0f}ms")
        if avg_q > best_avg:
            best_avg = avg_q
            best_n = n

best_n = best_n or 4
print(f"\n  Recommended top_N: {best_n}")
print(f"  → Set TOP_N = {best_n} in POC/core/jarvis_cli.py")

ts03 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out03 = EXPS_DIR / "experiment_03" / "results" / f"run_{ts03}_auto.json"
out03.parent.mkdir(exist_ok=True)
out03.write_text(json.dumps({
    "experiment": "exp_03_memory_count", "model": MODEL,
    "run_at": ts03, "recommended_top_n": best_n, "prompts": all_results03,
}, indent=2))
print(f"  Saved → {out03.name}")


# ── Experiments 04–07 — Run as subprocesses (fully automated) ────────────────

for exp_num, title in [
    ("04", "Extraction Prompt Strictness"),
    ("05", "TF-IDF vs Embeddings"),
    ("06", "Task Classification (Disinhibition Gate)"),
    ("07", "Graph vs Flat List"),
]:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_num} — {title}")
    print(f"{'='*60}")
    print(f"  Running...")
    output = run_subprocess(exp_num)
    # Print last 25 lines (summary)
    lines = [l for l in output.splitlines() if l.strip()]
    summary = "\n".join("  " + l for l in lines[-25:])
    print(summary)


# ── Experiment 08 — Run fresh + LLM judge ────────────────────────────────────

print(f"\n{'='*60}")
print("EXPERIMENT 08 — Confidence Decay: Does Weighting Help?")
print(f"{'='*60}")

# Build 100 memories (same as experiment_08/run.py)
high = [
    {"content": "user prefers direct answers without preamble",         "category": "preference", "confidence": 0.95},
    {"content": "user is building Project Jarvis in Python",            "category": "goal",       "confidence": 0.95},
    {"content": "user has 10 years of Python experience",               "category": "expertise",  "confidence": 0.90},
    {"content": "user uses MacBook Pro M3 Pro with 36GB RAM",           "category": "general",    "confidence": 0.95},
    {"content": "user wants Jarvis to work completely offline",         "category": "goal",       "confidence": 0.90},
    {"content": "user has Ollama running with llama3:latest",           "category": "general",    "confidence": 0.90},
    {"content": "user dislikes over-explained responses",               "category": "preference", "confidence": 0.90},
    {"content": "user prefers minimal Python dependencies",             "category": "preference", "confidence": 0.85},
    {"content": "user is testing PoC with SQLite and networkx",         "category": "general",    "confidence": 0.85},
    {"content": "user works in 2-hour morning focus blocks",            "category": "pattern",    "confidence": 0.85},
    {"content": "user prefers synchronous Python code for PoC",        "category": "preference", "confidence": 0.82},
    {"content": "user tracks tasks in Linear",                          "category": "pattern",    "confidence": 0.80},
    {"content": "user prefers Python type hints in all function sigs",  "category": "preference", "confidence": 0.83},
]
low = [
    {"content": "user prefers verbose detailed explanations (contradicted)", "category": "preference", "confidence": 0.15},
    {"content": "user once said they use Notion for tasks (superseded)", "category": "pattern",    "confidence": 0.20},
    {"content": "user might prefer async Python (contradicted)",         "category": "preference", "confidence": 0.20},
    {"content": "user may prefer working late at night (contradicted)",  "category": "pattern",    "confidence": 0.15},
]
padding = [
    {"content": f"user fact #{i} (filler)",                             "category": "general",    "confidence": 0.55}
    for i in range(83)
]
MEMS_08 = (high + low + padding)[:100]

TEST_CASES_08 = [
    {"prompt": "How should Jarvis format responses for this user?",
     "high_conf_answer": "user prefers direct answers without preamble",
     "low_conf_answer":  "user prefers verbose detailed explanations (contradicted)"},
    {"prompt": "What coding style does this user prefer for Python?",
     "high_conf_answer": "user prefers minimal Python dependencies",
     "low_conf_answer":  "user might prefer async Python (contradicted)"},
    {"prompt": "When should Jarvis avoid sending notifications?",
     "high_conf_answer": "user works in 2-hour morning focus blocks",
     "low_conf_answer":  "user may prefer working late at night (contradicted)"},
    {"prompt": "What task management tool does this user use?",
     "high_conf_answer": "user tracks tasks in Linear",
     "low_conf_answer":  "user once said they use Notion for tasks (superseded)"},
    {"prompt": "Should Jarvis explain things concisely or in detail?",
     "high_conf_answer": "user dislikes over-explained responses",
     "low_conf_answer":  "user prefers verbose detailed explanations (contradicted)"},
]

conn08 = fresh_db()
seed_memories(conn08, MEMS_08)
mems08 = get_memories(conn08)
hc = sum(1 for m in mems08 if m["confidence"] >= 0.8)
lc = sum(1 for m in mems08 if m["confidence"] < 0.45)
print(f"Seeded {len(mems08)} memories (high≥0.8: {hc}, low<0.45: {lc})\n")

all_results08 = []
auto_b_better = 0

def contains08(results, frag):
    return any(frag[:40].lower() in m["content"].lower() for m in results)

for i, test in enumerate(TEST_CASES_08, 1):
    prompt = test["prompt"]
    top_a = retrieve_tfidf(prompt, mems08, top_n=4, confidence_weight=False)
    top_b = retrieve_tfidf(prompt, mems08, top_n=4, confidence_weight=True)

    high_in_a = contains08(top_a, test["high_conf_answer"])
    low_in_a  = contains08(top_a, test["low_conf_answer"])
    high_in_b = contains08(top_b, test["high_conf_answer"])
    low_in_b  = contains08(top_b, test["low_conf_answer"])

    resp_a, lat_a = generate(build_prompt(prompt, top_a, fmt="structured"), model=MODEL, max_tokens=300)
    resp_b, lat_b = generate(build_prompt(prompt, top_b, fmt="structured"), model=MODEL, max_tokens=300)

    ra, rb = compare(prompt, resp_a, resp_b,
                     memories=[m["content"] for m in top_b[:2]], model=MODEL)

    # Automatic advantage: B suppressed contradicted facts
    if high_in_b and not low_in_b and (not high_in_a or low_in_a):
        auto_b_better += 1

    print(f"  Test {i}: A={ra} B={rb} | high_in_A={high_in_a} low_in_A={low_in_a} | "
          f"high_in_B={high_in_b} low_in_B={low_in_b}")
    print(f"    {prompt[:60]}")

    all_results08.append({
        "test_id": i, "prompt": prompt,
        "method_a": {"has_high_conf": high_in_a, "has_low_conf": low_in_a,
                     "response": resp_a, "latency_ms": lat_a,
                     "top_memories": [m["content"][:60] for m in top_a], "rating": ra},
        "method_b": {"has_high_conf": high_in_b, "has_low_conf": low_in_b,
                     "response": resp_b, "latency_ms": lat_b,
                     "top_memories": [m["content"][:60] for m in top_b], "rating": rb},
    })

b_better08 = sum(1 for r in all_results08 if r["method_b"]["rating"] > r["method_a"]["rating"])
a_better08 = sum(1 for r in all_results08 if r["method_a"]["rating"] > r["method_b"]["rating"])

if b_better08 >= 3 or auto_b_better >= 3:
    decision08 = "ACTIVATE — Set CONFIDENCE_WEIGHT = True in jarvis_cli.py"
else:
    decision08 = "DEFER — No clear benefit at 100 memories; leave CONFIDENCE_WEIGHT = False"

print(f"\n  B better (judge): {b_better08}/5 | Auto B advantage: {auto_b_better}/5")
print(f"  Decision: {decision08}")

ts08 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out08 = EXPS_DIR / "experiment_08" / "results" / f"run_{ts08}_auto.json"
out08.parent.mkdir(exist_ok=True)
out08.write_text(json.dumps({
    "experiment": "exp_08_confidence_decay", "model": MODEL, "run_at": ts08,
    "b_rated_better": b_better08, "a_rated_better": a_better08,
    "auto_b_better": auto_b_better, "decision": decision08,
    "results": all_results08,
}, indent=2))
print(f"  Saved → {out08.name}")


# ── Final Summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("FINAL SUMMARY — Recommended Configuration")
print(f"{'='*60}")

# Gather exp04-07 decisions from their latest result files
def get_decision(exp_num: str) -> str:
    jf = latest_json(exp_num)
    if not jf:
        return "no results"
    d = json.loads(jf.read_text())
    return d.get("decision", d.get("recommended", "see results"))

print(f"  Exp 01 — Memory injection:   {decision01}")
print(f"  Exp 02 — Injection format:   INJECTION_FORMAT = '{winner02}'")
print(f"  Exp 03 — Memory count:       TOP_N = {best_n}")
print(f"  Exp 04 — Extraction prompt:  {get_decision('04')}")
print(f"  Exp 05 — Retrieval method:   {get_decision('05')}")
print(f"  Exp 06 — Disinhibition gate: {get_decision('06')}")
print(f"  Exp 07 — Graph vs flat:      {get_decision('07')}")
print(f"  Exp 08 — Confidence weight:  {decision08}")

print(f"\n  Update POC/core/jarvis_cli.py:")
print(f"    TOP_N              = {best_n}")
print(f"    INJECTION_FORMAT   = '{winner02}'")
cw = "True" if ("ACTIVATE" in decision08) else "False"
print(f"    CONFIDENCE_WEIGHT  = {cw}")
print(f"\nAll results saved to POC/experiments/experiment_XX/results/\n")
