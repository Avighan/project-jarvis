"""
Experiment 14 — Adversarial Challenge Agent (Two-LLM Debate)
=============================================================
Question: Does an adversarial challenge agent (two-LLM debate) improve
          research and planning output quality?

Tests the architecture spec Section 7.7 — Adversarial Challenge Agent (Nova-style).

Method A (single-agent): generate response directly.
Method B (adversarial):
  Step 1 — generate initial response
  Step 2 — generate a critical challenge of that response
  Step 3 — synthesize an improved final answer addressing valid criticisms

Pass: Method B wins on >= 4/6 prompts by LLM judge
Decision printed at end, with breakdown by PLANNING vs RESEARCH.

Run: python3 POC/experiments/experiment_14/run.py
"""

import sys, json, pathlib, datetime, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from experiments.shared.judge  import compare
from core.working_memory        import build_prompt_no_memory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 6 test prompts (3 PLANNING + 3 RESEARCH) ─────────────────────────────────
TEST_PROMPTS = [
    {
        "type":   "PLANNING",
        "prompt": "Should I build the MCP server before or after getting to 500 real memories in production?",
    },
    {
        "type":   "PLANNING",
        "prompt": "How should I sequence Phase 2 and Phase 3 — memory engine hardening vs causal graph?",
    },
    {
        "type":   "PLANNING",
        "prompt": "What's the right decision framework for when to use graph retrieval vs flat TF-IDF?",
    },
    {
        "type":   "RESEARCH",
        "prompt": "What are the main risks in using a 7B model for structured JSON extraction at production scale?",
    },
    {
        "type":   "RESEARCH",
        "prompt": "What approaches exist for making LLM outputs more reliable for structured tasks without fine-tuning?",
    },
    {
        "type":   "RESEARCH",
        "prompt": "What are the tradeoffs between TF-IDF and embedding-based retrieval for a personal memory system?",
    },
]

CHALLENGE_PROMPT_TEMPLATE = """\
Challenge the following response. Find weaknesses, missing considerations, \
alternative approaches, or assumptions that may be wrong. Be direct and critical.

Original question: {prompt}

Response to challenge:
{initial_response}

Challenge (be specific):"""

SYNTHESIS_PROMPT_TEMPLATE = """\
You gave an initial response and received a challenge. Produce an improved final \
answer that addresses the valid criticisms while keeping what was good.

Original question: {prompt}

Your initial response:
{initial_response}

Challenge received:
{challenge}

Improved final answer:"""


def generate_adversarial(prompt: str, model: str) -> tuple[str, str, str, int, int, int]:
    """
    Run the 3-step adversarial pipeline.
    Returns: (initial_response, challenge, final_response, lat1_ms, lat2_ms, lat3_ms)
    """
    # Step 1: initial response
    initial, lat1 = generate(build_prompt_no_memory(prompt), model=model, max_tokens=400)

    # Step 2: challenge
    challenge_prompt = CHALLENGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        initial_response=initial,
    )
    challenge, lat2 = generate(challenge_prompt, model=model, max_tokens=300)

    # Step 3: synthesize improved answer
    synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        prompt=prompt,
        initial_response=initial,
        challenge=challenge,
    )
    final, lat3 = generate(synthesis_prompt, model=model, max_tokens=450)

    return initial, challenge, final, lat1, lat2, lat3


def run_experiment() -> None:
    model = pick_model()
    print(f"\n{'='*65}")
    print("Experiment 14 — Adversarial Challenge Agent (Two-LLM Debate)")
    print(f"Model: {model} | 6 prompts (3 PLANNING + 3 RESEARCH)")
    print(f"Note: Method B is ~3x slower due to 3 sequential LLM calls.")
    print(f"{'='*65}\n")

    # fresh_db() for isolation; seed minimal context memories for judge
    conn = fresh_db()
    base_memories = [
        {"content": "user is building Project Jarvis in Python 3.12",         "category": "goal",      "confidence": 0.95},
        {"content": "user is testing PoC with SQLite and networkx",            "category": "general",   "confidence": 0.85},
        {"content": "user wants Jarvis to work completely offline",            "category": "goal",      "confidence": 0.92},
        {"content": "user has 10 years Python experience; expert level",       "category": "expertise", "confidence": 0.92},
        {"content": "user prefers direct responses without preamble",          "category": "preference","confidence": 0.92},
        {"content": "user prefers minimal Python dependencies",                "category": "preference","confidence": 0.88},
        {"content": "user has Ollama running with llama3:latest locally",      "category": "general",   "confidence": 0.90},
        {"content": "user prefers synchronous Python code for PoC",           "category": "preference","confidence": 0.82},
    ]
    seed_memories(conn, base_memories)
    memories      = get_memories(conn)
    mem_contents  = [m["content"] for m in memories]

    print(f"Seeded {len(memories)} base context memories for judge.\n")

    all_results = []
    b_wins_total    = 0
    b_wins_planning = 0
    b_wins_research = 0

    for pi, test in enumerate(TEST_PROMPTS, 1):
        task_type = test["type"]
        prompt    = test["prompt"]
        print(f"Prompt {pi}/6 [{task_type}]: {prompt[:60]}...")

        # Method A: single-agent direct response
        t_a_start = time.time()
        resp_a, lat_a = generate(build_prompt_no_memory(prompt), model=model, max_tokens=400)
        t_a_total = int((time.time() - t_a_start) * 1000)

        # Method B: adversarial 3-step
        t_b_start = time.time()
        initial_b, challenge_b, resp_b, lat_b1, lat_b2, lat_b3 = generate_adversarial(
            prompt, model=model
        )
        t_b_total = int((time.time() - t_b_start) * 1000)

        # LLM judge compares A vs B final responses
        score_a, score_b = compare(prompt, resp_a, resp_b, mem_contents, model=model)

        winner = "B" if score_b > score_a else ("A" if score_a > score_b else "tie")
        if winner == "B":
            b_wins_total += 1
            if task_type == "PLANNING":
                b_wins_planning += 1
            else:
                b_wins_research += 1

        len_a = len(resp_a.split())
        len_b = len(resp_b.split())

        print(f"  Judge scores — A:{score_a}  B:{score_b}  winner={winner}")
        print(f"  Latency — A:{t_a_total}ms  B:{t_b_total}ms  (speedup factor: {t_b_total/max(t_a_total,1):.1f}x)")
        print(f"  Response length (words) — A:{len_a}  B:{len_b}")

        all_results.append({
            "prompt_id":    pi,
            "type":         task_type,
            "prompt":       prompt,
            "winner":       winner,
            "method_a": {
                "response":    resp_a,
                "latency_ms":  t_a_total,
                "length_words": len_a,
                "judge_score": score_a,
            },
            "method_b": {
                "initial_response": initial_b,
                "challenge":        challenge_b,
                "final_response":   resp_b,
                "latency_ms":       t_b_total,
                "latency_step1_ms": lat_b1,
                "latency_step2_ms": lat_b2,
                "latency_step3_ms": lat_b3,
                "length_words":     len_b,
                "judge_score":      score_b,
            },
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    planning_results = [r for r in all_results if r["type"] == "PLANNING"]
    research_results = [r for r in all_results if r["type"] == "RESEARCH"]

    avg_a_overall = sum(r["method_a"]["judge_score"] for r in all_results) / len(all_results)
    avg_b_overall = sum(r["method_b"]["judge_score"] for r in all_results) / len(all_results)

    avg_a_planning = sum(r["method_a"]["judge_score"] for r in planning_results) / len(planning_results)
    avg_b_planning = sum(r["method_b"]["judge_score"] for r in planning_results) / len(planning_results)
    avg_a_research = sum(r["method_a"]["judge_score"] for r in research_results) / len(research_results)
    avg_b_research = sum(r["method_b"]["judge_score"] for r in research_results) / len(research_results)

    avg_lat_a = sum(r["method_a"]["latency_ms"] for r in all_results) / len(all_results)
    avg_lat_b = sum(r["method_b"]["latency_ms"] for r in all_results) / len(all_results)
    latency_factor = avg_lat_b / max(avg_lat_a, 1)

    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"Method B wins overall:  {b_wins_total}/6")
    print(f"  PLANNING:             {b_wins_planning}/3")
    print(f"  RESEARCH:             {b_wins_research}/3")
    print(f"Avg judge score A:      {avg_a_overall:.2f}")
    print(f"Avg judge score B:      {avg_b_overall:.2f}")
    print(f"  PLANNING  A:{avg_a_planning:.2f}  B:{avg_b_planning:.2f}")
    print(f"  RESEARCH  A:{avg_a_research:.2f}  B:{avg_b_research:.2f}")
    print(f"Avg latency A:          {avg_lat_a:.0f}ms")
    print(f"Avg latency B:          {avg_lat_b:.0f}ms  ({latency_factor:.1f}x slower)")

    if b_wins_total >= 4:
        if b_wins_planning >= 2 and b_wins_research >= 2:
            decision = (
                f"PASS — adversarial agent improves output quality on {b_wins_total}/6 prompts; "
                f"build into Orchestration Engine for RESEARCH/PLANNING tasks "
                f"(PLANNING: {b_wins_planning}/3, RESEARCH: {b_wins_research}/3)"
            )
        elif b_wins_research >= 2:
            decision = (
                f"PARTIAL — improvement on RESEARCH ({b_wins_research}/3) "
                f"but not PLANNING ({b_wins_planning}/3); "
                f"consider adversarial agent for RESEARCH tasks only"
            )
        elif b_wins_planning >= 2:
            decision = (
                f"PARTIAL — improvement on PLANNING ({b_wins_planning}/3) "
                f"but not RESEARCH ({b_wins_research}/3); "
                f"consider adversarial agent for PLANNING tasks only"
            )
        else:
            decision = (
                f"PASS (overall) — adversarial agent wins {b_wins_total}/6 prompts; "
                f"latency factor {latency_factor:.1f}x — validate at production scale"
            )
    else:
        decision = (
            f"FAIL — {latency_factor:.1f}x latency cost not justified by quality gain "
            f"(B wins only {b_wins_total}/6 prompts)"
        )

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":       "exp_14_adversarial_challenge_agent",
        "model":            model,
        "run_at":           ts,
        "b_wins_total":     b_wins_total,
        "b_wins_planning":  b_wins_planning,
        "b_wins_research":  b_wins_research,
        "avg_score_a":      round(avg_a_overall, 3),
        "avg_score_b":      round(avg_b_overall, 3),
        "avg_score_a_planning": round(avg_a_planning, 3),
        "avg_score_b_planning": round(avg_b_planning, 3),
        "avg_score_a_research": round(avg_a_research, 3),
        "avg_score_b_research": round(avg_b_research, 3),
        "avg_latency_a_ms": round(avg_lat_a),
        "avg_latency_b_ms": round(avg_lat_b),
        "latency_factor":   round(latency_factor, 2),
        "decision":         decision,
        "results":          all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 14 — Adversarial Challenge Agent\n"
        f"Run: {ts} | Model: {model}\n"
        f"B wins: {b_wins_total}/6 (PLANNING: {b_wins_planning}/3, RESEARCH: {b_wins_research}/3)\n"
        f"Avg score A: {avg_a_overall:.2f} | Avg score B: {avg_b_overall:.2f}\n"
        f"  PLANNING A:{avg_a_planning:.2f} B:{avg_b_planning:.2f}\n"
        f"  RESEARCH A:{avg_a_research:.2f} B:{avg_b_research:.2f}\n"
        f"Latency factor: {latency_factor:.1f}x\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
