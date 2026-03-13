"""
Experiment 6 — Disinhibition: Does Task Classification Work?
=============================================================
Tests whether llama3 can reliably classify queries into 6 task types.
30 prompts (5 per type). Measures accuracy and confusion patterns.
Pass criterion: ≥80% overall accuracy.

Run: python3 POC/experiments/experiment_06/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CATEGORIES = ["PLANNING", "RESEARCH", "EXECUTION", "EMOTIONAL", "REFLECTION", "LEARNING"]

# 30 test prompts — 5 per category — with ground truth labels
LABELED_PROMPTS = [
    # ── PLANNING (5) ──────────────────────────────────────────────────────────
    {"prompt": "Help me structure my work for this week given I have 3 deadlines.",
     "label": "PLANNING"},
    {"prompt": "What should I prioritise between the memory graph and the MCP server?",
     "label": "PLANNING"},
    {"prompt": "Create a 2-week sprint plan for the Jarvis PoC.",
     "label": "PLANNING"},
    {"prompt": "How should I sequence the 8 PoC experiments?",
     "label": "PLANNING"},
    {"prompt": "I need to plan my month — what should I tackle first?",
     "label": "PLANNING"},
    # ── RESEARCH (5) ─────────────────────────────────────────────────────────
    {"prompt": "What do I know about transformer attention from previous research?",
     "label": "RESEARCH"},
    {"prompt": "Find me information about Granger causality methods in 2024.",
     "label": "RESEARCH"},
    {"prompt": "What's the current state of local LLM memory implementations?",
     "label": "RESEARCH"},
    {"prompt": "Summarise what I've read about STABLE gating for LoRA.",
     "label": "RESEARCH"},
    {"prompt": "What approaches exist for continual learning in neural networks?",
     "label": "RESEARCH"},
    # ── EXECUTION (5) ────────────────────────────────────────────────────────
    {"prompt": "Create a Linear ticket for the SQLite schema migration bug.",
     "label": "EXECUTION"},
    {"prompt": "Write the Python code for the memory write-back function.",
     "label": "EXECUTION"},
    {"prompt": "Send a summary of today's progress to my notes.",
     "label": "EXECUTION"},
    {"prompt": "Run the experiment_01 script and show me the output.",
     "label": "EXECUTION"},
    {"prompt": "Create a new file called retrieval_v2.py with the BM25 implementation.",
     "label": "EXECUTION"},
    # ── EMOTIONAL (5) ────────────────────────────────────────────────────────
    {"prompt": "I'm feeling really overwhelmed with everything on my plate right now.",
     "label": "EMOTIONAL"},
    {"prompt": "I'm frustrated that the memory injection isn't working as expected.",
     "label": "EMOTIONAL"},
    {"prompt": "Had a great day — everything clicked and the PoC is coming together.",
     "label": "EMOTIONAL"},
    {"prompt": "I'm worried I'm building the wrong thing and wasting time.",
     "label": "EMOTIONAL"},
    {"prompt": "Feeling burnt out. Not sure how to get my motivation back.",
     "label": "EMOTIONAL"},
    # ── REFLECTION (5) ────────────────────────────────────────────────────────
    {"prompt": "How have my coding habits changed over the last month?",
     "label": "REFLECTION"},
    {"prompt": "Looking back at this week, what went well and what didn't?",
     "label": "REFLECTION"},
    {"prompt": "What patterns have I noticed in how I approach architecture decisions?",
     "label": "REFLECTION"},
    {"prompt": "Have my priorities shifted since I started working on Jarvis?",
     "label": "REFLECTION"},
    {"prompt": "What have I learned from the first 3 PoC experiments?",
     "label": "REFLECTION"},
    # ── LEARNING (5) ─────────────────────────────────────────────────────────
    {"prompt": "Explain how Granger causality works and when it applies.",
     "label": "LEARNING"},
    {"prompt": "How does TF-IDF differ from BM25 scoring?",
     "label": "LEARNING"},
    {"prompt": "Teach me the basics of the Ollama embeddings API.",
     "label": "LEARNING"},
    {"prompt": "What is the VIP-cell 'brake on the brake' mechanism in neuroscience?",
     "label": "LEARNING"},
    {"prompt": "How does LoRA fine-tuning work at a high level?",
     "label": "LEARNING"},
]

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


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 6 — Task Classification Accuracy")
    print(f"Model: {model} | {len(LABELED_PROMPTS)} prompts × {len(CATEGORIES)} categories")
    print(f"{'='*60}\n")

    results = []
    confusion = {true: {pred: 0 for pred in CATEGORIES} for true in CATEGORIES}

    for i, item in enumerate(LABELED_PROMPTS, 1):
        prompt     = item["prompt"]
        true_label = item["label"]

        full_prompt = CLASSIFICATION_PROMPT.format(prompt=prompt)
        raw, latency_ms = generate(full_prompt, model=model, temperature=0.0, max_tokens=20)

        # Parse predicted label
        predicted = raw.strip().upper()
        matched = next((c for c in CATEGORIES if c in predicted), None)
        if not matched:
            matched = "UNKNOWN"

        correct = (matched == true_label)
        results.append({
            "prompt":    prompt,
            "true":      true_label,
            "predicted": matched,
            "raw":       raw.strip(),
            "correct":   correct,
            "latency_ms": latency_ms,
        })

        if matched in CATEGORIES:
            confusion[true_label][matched] += 1

        symbol = "✓" if correct else "✗"
        print(f"  {symbol} [{i:2d}] True={true_label:10s} Pred={matched:10s} "
              f"({latency_ms}ms) | {prompt[:45]}...")

    # ── Metrics ──────────────────────────────────────────────────────────────
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = total_correct / len(results)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Overall accuracy: {accuracy:.0%} ({total_correct}/{len(results)})")

    print("\nPer-class accuracy:")
    per_class = {}
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["true"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"])
        cat_acc     = cat_correct / len(cat_results)
        per_class[cat] = cat_acc
        print(f"  {cat:12s}: {cat_acc:.0%} ({cat_correct}/5)")

    # Common confusions
    print("\nCommon confusions:")
    for true_cat in CATEGORIES:
        for pred_cat in CATEGORIES:
            if true_cat != pred_cat and confusion[true_cat][pred_cat] > 0:
                print(f"  {true_cat} → misclassified as {pred_cat}: "
                      f"{confusion[true_cat][pred_cat]}x")

    # Decision
    if accuracy >= 0.80:
        decision = f"PASS ({accuracy:.0%} ≥ 80%) — disinhibition model is viable. " \
                   f"Proceed with task classification in retrieval.py."
    elif accuracy >= 0.65:
        decision = (f"PARTIAL ({accuracy:.0%}) — viable with fallback rules. "
                    f"Add rules for confused pairs to retrieval.py.")
    else:
        decision = (f"FAIL ({accuracy:.0%} < 65%) — llama3 classification unreliable. "
                    f"Use simpler retrieval for PoC; revisit in Phase 2.")

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":  "exp_06_task_classification",
        "model":       model,
        "run_at":      ts,
        "accuracy":    accuracy,
        "per_class":   per_class,
        "confusion":   confusion,
        "decision":    decision,
        "results":     results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 6 — Task Classification\n"
        f"Run: {ts} | Model: {model}\n"
        f"Overall accuracy: {accuracy:.0%}\n"
        f"Per-class: { {k: f'{v:.0%}' for k,v in per_class.items()} }\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
