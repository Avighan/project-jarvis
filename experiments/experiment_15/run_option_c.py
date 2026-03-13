"""
Experiment 15b — Option C: Summary Paragraph Extraction
========================================================
Instead of forcing structured JSON, ask the model to write a brief
summary paragraph of what it learned about the user, then store as
a single episodic memory.

Hypothesis: removing schema compliance pressure improves recall and
eliminates hallucinations, at the cost of losing category structure.

Scoring: count how many expected facts appear in the summary paragraph.
Baseline: llama3 single-step = 69% recall, 1 hallucination.

Run: python3 POC/experiments/experiment_15/run_option_c.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from core.extractor import format_conversation

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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
        "expected_facts": [
            "prefers minimal schemas",
            "dislikes over-engineering",
            "wants code directly without preamble",
            "building Project Jarvis personal AI memory system",
        ]
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
        "expected_facts": [
            "works in 2-hour focused morning blocks",
            "prove memory injection works before building more",
            "senior Python engineer 10 years experience",
            "focused morning work sessions",
        ]
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
        "expected_facts": [
            "zero extra dependencies for PoC",
            "use numpy not sklearn",
            "knows TF-IDF basics",
            "has used BM25 in Java",
        ]
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
        "expected_facts": [
            "needs 3+ uninterrupted hours for architecture work",
            "meeting-heavy weeks reduce deep work",
        ]
    },
    {
        "id": 5,
        "description": "Edge case: ambiguous conversation, minimal extractable facts",
        "turns": [
            {"role": "user",      "content": "What time is it in London?"},
            {"role": "assistant", "content": "It's currently 3pm in London (GMT)."},
            {"role": "user",      "content": "Thanks."},
        ],
        "expected_facts": []
    },
]

SUMMARY_PROMPT = """\
Read this conversation and write a brief factual summary of what you learned about the USER.
Focus only on standing facts: preferences, skills, goals, work patterns, habits.
Do NOT include things the assistant said. Do NOT include momentary questions.
If nothing meaningful can be learned about the user, write: Nothing significant to note.
Write 2-4 sentences maximum. Be specific and factual.

Conversation:
{conversation}

Summary of what I learned about this user:"""


def score_summary(summary: str, expected_facts: list[str]) -> dict:
    summary_lower = summary.lower()
    expected_count = len(expected_facts)
    if expected_count == 0:
        return {"expected": 0, "found": 0, "recall": None}
    found = sum(
        1 for fact in expected_facts
        if any(word in summary_lower for word in fact.lower().split() if len(word) > 4)
    )
    return {
        "expected": expected_count,
        "found":    found,
        "recall":   found / expected_count,
    }


def check_hallucinations(summary: str, conversation_text: str) -> bool:
    """Rough check: are there significant words in summary not rooted in conversation?"""
    summary_words = [w for w in summary.lower().split() if len(w) > 6]
    conv_lower = conversation_text.lower()
    if not summary_words:
        return False
    miss = sum(1 for w in summary_words if w[:5] not in conv_lower)
    return (miss / len(summary_words)) > 0.4


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 15b — Option C: Summary Paragraph Extraction")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    all_results = []
    all_recalls = []

    for conv in CONVERSATIONS:
        print(f"\nConversation {conv['id']}: {conv['description']}")
        conversation_text = format_conversation(conv["turns"])

        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
        summary, latency = generate(prompt, model=model, temperature=0.1, max_tokens=200)
        summary = summary.strip()

        print(f"  Summary ({latency}ms): {summary}")

        metrics = score_summary(summary, conv["expected_facts"])
        hallucinated = check_hallucinations(summary, conversation_text)

        if metrics["recall"] is not None:
            print(f"  Score: expected={metrics['expected']} | found={metrics['found']} | recall={metrics['recall']:.0%}")
            all_recalls.append(metrics["recall"])
        else:
            print("  Score: no expected facts (edge case)")

        if hallucinated:
            print("  Possible hallucination detected in summary.")
        else:
            print("  No hallucinations detected.")

        all_results.append({
            "conversation_id": conv["id"],
            "description":     conv["description"],
            "summary":         summary,
            "latency_ms":      latency,
            "metrics":         metrics,
            "hallucinated":    hallucinated,
        })

    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")

    avg_recall  = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    total_hall  = sum(1 for r in all_results if r["hallucinated"])

    print(f"Average recall:       {avg_recall:.0%}  (Exp 04 baseline: 69%)")
    print(f"Hallucinated convs:   {total_hall}/5   (Exp 04 baseline: 1 item)")

    if avg_recall > 0.75 and total_hall == 0:
        decision = "PASS — Option C beats baseline on recall with zero hallucinations"
    elif avg_recall > 0.69:
        decision = "PARTIAL — Option C improves recall; acceptable for manual seeding phase"
    else:
        decision = "FAIL — Option C does not improve over baseline; proceed to Option D (Claude API)"

    print(f"\nDecision: {decision}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":    "exp_15b_option_c_summary_extraction",
        "model":         model,
        "run_at":        ts,
        "avg_recall":    avg_recall,
        "total_hallucinated_convs": total_hall,
        "decision":      decision,
        "conversations": all_results,
    }
    json_path = RESULTS_DIR / f"run_optionC_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {json_path}")
    return avg_recall, total_hall, decision


if __name__ == "__main__":
    run_experiment()
