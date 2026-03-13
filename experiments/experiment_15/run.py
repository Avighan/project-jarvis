"""
Experiment 15 — Two-Step Extraction Quality
============================================
Does a two-step approach beat single-step extraction (Exp 04)?

Step 1: Extract candidate facts as free prose (low precision requirement).
Step 2: For each candidate, ask YES/NO — "Is this actually stated in the conversation?"

Uses the same 5 conversations and ground truth as Exp 04 for direct comparison.
Baseline: llama3 single-step = 69% recall, 1 hallucination.

Run: python3 POC/experiments/experiment_15/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from core.extractor import format_conversation

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Same ground truth as Exp 04 ──────────────────────────────────────────────
CONVERSATIONS = [
    {
        "id": 1,
        "description": "Technical discussion revealing preferences and project context",
        "turns": [
            {"role": "user",      "content": "Can you help me design the SQLite schema for storing user memories?"},
            {"role": "assistant", "content": "Sure, what constraints do you have?"},
            {"role": "user",      "content": "It needs to support confidence scoring and typed edges between memories. "
                                             "I'm building this for Project Jarvis — my personal AI memory system. "
                                             "Keep the schema minimal, I don't like overengineering things."},
            {"role": "assistant", "content": "Got it. Here's a minimal schema..."},
            {"role": "user",      "content": "Perfect. And please skip the preamble next time, "
                                             "just give me the code directly."},
        ],
        "expected": {
            "preferences": ["prefers minimal schemas", "dislikes over-engineering",
                            "wants code directly without preamble"],
            "goals":       ["building Project Jarvis — personal AI memory system"],
            "expertise":   [],
            "patterns":    [],
        }
    },
    {
        "id": 2,
        "description": "Planning conversation revealing expertise and work style",
        "turns": [
            {"role": "user",      "content": "I need to think through the PoC timeline for Jarvis. "
                                             "I'm a senior Python engineer, about 10 years in, "
                                             "so I can move fast on the implementation side."},
            {"role": "assistant", "content": "Great, what's the biggest risk you're trying to derisk first?"},
            {"role": "user",      "content": "Memory injection — does it actually improve responses? "
                                             "I want to prove that before building anything else. "
                                             "I usually work in 2-hour focused blocks in the morning."},
        ],
        "expected": {
            "preferences": ["works in 2-hour focused morning blocks"],
            "goals":       ["prove memory injection works before building more"],
            "expertise":   ["senior Python engineer, ~10 years experience"],
            "patterns":    ["focused morning work sessions"],
        }
    },
    {
        "id": 3,
        "description": "Research conversation with domain knowledge signals",
        "turns": [
            {"role": "user",      "content": "What do you know about TF-IDF retrieval? "
                                             "I know the basics — term frequency, inverse document frequency — "
                                             "but I'm fuzzy on the IDF weighting when the corpus is small."},
            {"role": "assistant", "content": "With a small corpus, IDF values become noisy..."},
            {"role": "user",      "content": "Right, I thought so. I've used BM25 before on a search project "
                                             "but that was in Java. I'll stick to numpy for this PoC "
                                             "since I want zero extra dependencies."},
        ],
        "expected": {
            "preferences": ["zero extra dependencies for PoC", "use numpy not sklearn"],
            "goals":       [],
            "expertise":   ["knows TF-IDF basics", "has used BM25 in Java"],
            "patterns":    [],
        }
    },
    {
        "id": 4,
        "description": "Personal conversation with minimal extractable technical facts",
        "turns": [
            {"role": "user",      "content": "Had a rough week. Too many meetings, hard to get deep work done."},
            {"role": "assistant", "content": "That sounds frustrating. What kind of deep work did you miss?"},
            {"role": "user",      "content": "Mostly the Jarvis coding. I need at least 3 uninterrupted hours "
                                             "to make real progress on architecture problems."},
        ],
        "expected": {
            "preferences": [],
            "goals":       [],
            "expertise":   [],
            "patterns":    ["needs 3+ uninterrupted hours for architecture work",
                            "meeting-heavy weeks reduce deep work"],
        }
    },
    {
        "id": 5,
        "description": "Edge case: ambiguous conversation, minimal extractable facts",
        "turns": [
            {"role": "user",      "content": "What time is it in London?"},
            {"role": "assistant", "content": "It's currently 3pm in London (GMT)."},
            {"role": "user",      "content": "Thanks."},
        ],
        "expected": {
            "preferences": [],
            "goals":       [],
            "expertise":   [],
            "patterns":    [],
        }
    },
]

# ── Step 1: Extract candidates as prose ─────────────────────────────────────

STEP1_PROMPT = """\
Read this conversation and list every standing fact about the USER.
Standing facts = things persistently true about who the user is: preferences, skills, goals, habits, work patterns.
Do NOT include things the assistant said. Do NOT include momentary questions or one-off actions.

Write each fact on its own line, starting with a dash (-).
If nothing is extractable, write: NONE

Conversation:
{conversation}

Facts about the user:"""


def step1_extract_prose(conversation: str, model: str) -> list[str]:
    """Step 1: Get candidate facts as a prose list."""
    prompt = STEP1_PROMPT.format(conversation=conversation)
    raw, latency = generate(prompt, model=model, temperature=0.1, max_tokens=400)

    candidates = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("-"):
            fact = line.lstrip("- ").strip()
            if fact:
                candidates.append(fact)
        elif line.upper() == "NONE":
            break
    return candidates, latency


# ── Step 2: Validate each candidate ─────────────────────────────────────────

STEP2_PROMPT = """\
Conversation:
{conversation}

Candidate fact about the user: "{fact}"

Is this fact actually stated or clearly implied by the USER's own words in the conversation above?
Answer with exactly one word: YES or NO"""


def step2_validate(fact: str, conversation: str, model: str) -> tuple[bool, int]:
    """Step 2: YES/NO validation of a single candidate fact."""
    prompt = STEP2_PROMPT.format(conversation=conversation, fact=fact)
    raw, latency = generate(prompt, model=model, temperature=0.0, max_tokens=5)
    answer = raw.strip().upper()
    return answer.startswith("YES"), latency


# ── Scoring (same as Exp 04) ─────────────────────────────────────────────────

def score_facts(validated_facts: list[str], expected: dict) -> dict:
    """Score validated facts against ground truth across all categories."""
    all_expected = []
    for key in ["preferences", "goals", "expertise", "patterns"]:
        all_expected.extend(expected.get(key, []))

    expected_count = len(all_expected)
    extracted_count = len(validated_facts)
    validated_lower = [f.lower() for f in validated_facts]

    found = sum(
        1 for exp in all_expected
        if any(
            any(word in ext for word in exp.lower().split() if len(word) > 4)
            for ext in validated_lower
        )
    )
    return {
        "expected":  expected_count,
        "extracted": extracted_count,
        "found":     found,
        "recall":    found / expected_count if expected_count > 0 else None,
    }


def check_hallucinations(validated_facts: list[str], conversation_text: str) -> list[str]:
    conv_lower = conversation_text.lower()
    hallucinations = []
    for fact in validated_facts:
        sig_words = [w for w in fact.lower().split() if len(w) > 5]
        if sig_words:
            def word_in_conv(w):
                if w in conv_lower:
                    return True
                return w[:5] in conv_lower
            miss_rate = sum(1 for w in sig_words if not word_in_conv(w)) / len(sig_words)
            if miss_rate > 0.5:
                hallucinations.append(fact)
    return hallucinations


# ── Main ─────────────────────────────────────────────────────────────────────

def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 15 — Two-Step Extraction Quality")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    all_results = []
    all_recalls = []

    for conv in CONVERSATIONS:
        print(f"\nConversation {conv['id']}: {conv['description']}")
        conversation_text = format_conversation(conv["turns"])

        # Step 1
        candidates, latency1 = step1_extract_prose(conversation_text, model)
        print(f"  Step 1 candidates ({latency1}ms): {candidates}")

        # Step 2
        validated = []
        rejected  = []
        total_latency2 = 0
        for candidate in candidates:
            passed, lat = step2_validate(candidate, conversation_text, model)
            total_latency2 += lat
            if passed:
                validated.append(candidate)
            else:
                rejected.append(candidate)

        print(f"  Step 2 validated ({total_latency2}ms): {validated}")
        if rejected:
            print(f"  Step 2 rejected: {rejected}")

        # Score
        metrics = score_facts(validated, conv["expected"])
        hallucinations = check_hallucinations(validated, conversation_text)

        print(f"  Score: expected={metrics['expected']} | extracted={metrics['extracted']} | "
              f"found={metrics['found']} | recall={metrics['recall']:.0%}" if metrics['recall'] is not None
              else f"  Score: no expected facts (edge case)")
        if hallucinations:
            print(f"  Possible hallucinations: {hallucinations}")
        else:
            print("  No hallucinations detected.")

        if metrics["recall"] is not None:
            all_recalls.append(metrics["recall"])

        all_results.append({
            "conversation_id":  conv["id"],
            "description":      conv["description"],
            "candidates":       candidates,
            "validated":        validated,
            "rejected":         rejected,
            "latency_step1_ms": latency1,
            "latency_step2_ms": total_latency2,
            "metrics":          metrics,
            "hallucinations":   hallucinations,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")

    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    total_hall = sum(len(r["hallucinations"]) for r in all_results)
    total_candidates = sum(len(r["candidates"]) for r in all_results)
    total_validated  = sum(len(r["validated"])  for r in all_results)
    rejection_rate = 1 - (total_validated / total_candidates) if total_candidates > 0 else 0

    print(f"Average recall:          {avg_recall:.0%}  (Exp 04 baseline: 69%)")
    print(f"Total hallucinations:    {total_hall}      (Exp 04 baseline: 1)")
    print(f"Total candidates:        {total_candidates}")
    print(f"Total validated:         {total_validated}")
    print(f"Step 2 rejection rate:   {rejection_rate:.0%}")

    # Decision vs baseline
    baseline_recall = 0.69
    baseline_hall   = 1

    if avg_recall > baseline_recall and total_hall <= baseline_hall:
        decision = "PASS — two-step beats single-step on both recall and hallucinations"
    elif avg_recall > baseline_recall:
        decision = "PARTIAL — two-step improves recall but more hallucinations"
    elif total_hall < baseline_hall:
        decision = "PARTIAL — two-step reduces hallucinations but recall unchanged"
    else:
        decision = "FAIL — two-step does not improve over single-step baseline"

    print(f"\nDecision: {decision}")

    # Save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":       "exp_15_two_step_extraction",
        "model":            model,
        "run_at":           ts,
        "avg_recall":       avg_recall,
        "baseline_recall":  baseline_recall,
        "total_hallucinations": total_hall,
        "baseline_hallucinations": baseline_hall,
        "rejection_rate":   rejection_rate,
        "decision":         decision,
        "conversations":    all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
