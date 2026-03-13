"""
Experiment 4 — Implicit Extraction Quality
==========================================
Can llama3 reliably extract structured facts from conversations?
Tests extraction prompt against 5 pre-written conversation excerpts.
Measures: accuracy, hallucinations, JSON parse rate, schema consistency.

Run: python3 POC/experiments/experiment_04/run.py
"""

import sys, json, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from core.extractor             import extract_facts, format_conversation

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Ground truth conversations + expected extractions ────────────────────────
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


def score_extraction(extracted: dict, expected: dict) -> dict:
    """Compare extracted facts to expected ground truth. Returns precision/recall metrics."""
    metrics = {}
    for key in ["preferences", "goals", "expertise", "patterns"]:
        expected_items = expected.get(key, [])
        extracted_items = extracted.get(key, [])
        extracted_texts = [
            (item.get("fact") or item.get("goal") or item.get("topic") or item.get("pattern") or "").lower()
            for item in extracted_items
        ]
        expected_count = len(expected_items)
        extracted_count = len(extracted_items)
        # Rough overlap check: does each expected item appear in any extracted item?
        found = sum(
            1 for exp in expected_items
            if any(
                any(word in ext for word in exp.lower().split() if len(word) > 4)
                for ext in extracted_texts
            )
        )
        metrics[key] = {
            "expected":  expected_count,
            "extracted": extracted_count,
            "found":     found,
            "recall":    found / expected_count if expected_count > 0 else None,
        }
    return metrics


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 4 — Implicit Extraction Quality")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    all_results = []

    for conv in CONVERSATIONS:
        print(f"\nConversation {conv['id']}: {conv['description']}")
        conversation_text = format_conversation(conv["turns"])
        extracted = extract_facts(conversation_text, model=model)

        parse_ok = "_parse_failed" not in extracted
        latency  = extracted.pop("_latency_ms", None)
        raw      = extracted.pop("_raw", "")
        extracted.pop("_parse_success", None)

        # Score against ground truth
        metrics = score_extraction(extracted, conv["expected"])

        # Check for hallucinations (extracted items not in conversation)
        # Uses substring matching to handle morphological variants
        # (e.g. "directly" in conv matches "direct" in extraction)
        conv_text_lower = conversation_text.lower()
        hallucinations = []
        for key in ["preferences", "goals", "expertise", "patterns"]:
            for item in extracted.get(key, []):
                item_text = (item.get("fact") or item.get("goal") or
                             item.get("topic") or item.get("pattern") or "")
                # Flag if >50% of significant words have no root form in conversation
                sig_words = [w for w in item_text.lower().split() if len(w) > 5]
                if sig_words:
                    # A word is "found" if it appears as a substring of the conversation,
                    # or if the conversation contains a word that starts with it (stem match)
                    def word_in_conv(w):
                        if w in conv_text_lower:
                            return True
                        # Check if any conversation word starts with the first 5 chars (stem)
                        stem = w[:5]
                        return stem in conv_text_lower
                    miss_rate = sum(1 for w in sig_words if not word_in_conv(w)) / len(sig_words)
                    if miss_rate > 0.5:
                        hallucinations.append(item_text)

        result = {
            "conversation_id":  conv["id"],
            "description":      conv["description"],
            "parse_success":    parse_ok,
            "latency_ms":       latency,
            "extracted":        extracted,
            "expected":         conv["expected"],
            "metrics":          metrics,
            "hallucinations":   hallucinations,
        }
        all_results.append(result)

        # Print summary
        print(f"  Parse success:    {parse_ok}")
        print(f"  Latency:          {latency}ms")
        for key, m in metrics.items():
            if m["expected"] > 0:
                print(f"  {key:12s}: expected={m['expected']} | extracted={m['extracted']} | "
                      f"found={m['found']} | recall={m['recall']:.0%}")
        if hallucinations:
            print(f"  ⚠ Possible hallucinations: {hallucinations}")
        else:
            print("  No hallucinations detected.")

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")
    parse_rate = sum(1 for r in all_results if r["parse_success"]) / len(all_results)
    total_hall = sum(len(r["hallucinations"]) for r in all_results)
    print(f"JSON parse success rate: {parse_rate:.0%} ({sum(1 for r in all_results if r['parse_success'])}/{len(all_results)})")
    print(f"Total hallucinated items: {total_hall}")

    all_recalls = []
    for r in all_results:
        for key, m in r["metrics"].items():
            if m["recall"] is not None:
                all_recalls.append(m["recall"])
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    print(f"Average recall:          {avg_recall:.0%}")

    # Decision
    if parse_rate >= 0.8 and avg_recall >= 0.7 and total_hall == 0:
        decision = "PASS — extraction prompt works well on this model"
    elif parse_rate >= 0.6:
        decision = "PARTIAL — add JSON repair post-processing; tighten schema"
    else:
        decision = "FAIL — model cannot reliably output structured JSON; add few-shot examples"

    print(f"\nDecision: {decision}")
    print("\nRecommended fix if parse_rate < 80%:")
    print("  Add 1 few-shot example to the extraction prompt in extractor.py")
    print("  Or add json.loads() repair logic in _parse_json_response()")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":     "exp_04_extraction_quality",
        "model":          model,
        "run_at":         ts,
        "parse_rate":     parse_rate,
        "avg_recall":     avg_recall,
        "total_hallucinations": total_hall,
        "decision":       decision,
        "conversations":  all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 4 — Extraction Quality\n"
        f"Run: {ts} | Model: {model}\n"
        f"Parse rate: {parse_rate:.0%} | Avg recall: {avg_recall:.0%} | "
        f"Hallucinations: {total_hall}\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
