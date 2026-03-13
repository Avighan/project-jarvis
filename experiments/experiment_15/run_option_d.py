"""
Experiment 15c — Option D: Claude API Extraction
=================================================
Uses claude-haiku-4-5 via Anthropic API for session-end fact extraction.
Same conversations and ground truth as Exp 04 for direct comparison.

Baseline: llama3 single-step = 69% recall, 1 hallucination, 100% parse rate.
Cost: ~$0.001/session (Haiku pricing).

Run: python3 POC/experiments/experiment_15/run_option_d.py
"""

import sys, json, pathlib, datetime, os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import anthropic
from core.extractor import EXTRACTION_SCHEMA, EXTRACTION_PROMPT_TEMPLATE, _parse_json_response, format_conversation

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL   = "claude-haiku-4-5-20251001"

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


def extract_with_claude(conversation: str, client: anthropic.Anthropic) -> tuple[dict, int, int]:
    """Run extraction via Claude API. Returns (parsed_dict, input_tokens, output_tokens)."""
    import time
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        schema=EXTRACTION_SCHEMA,
        conversation=conversation.strip()
    )
    start = time.time()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=800,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.time() - start) * 1000)
    raw = msg.content[0].text
    parsed = _parse_json_response(raw)
    parsed["_raw"] = raw
    parsed["_latency_ms"] = latency_ms
    parsed["_input_tokens"]  = msg.usage.input_tokens
    parsed["_output_tokens"] = msg.usage.output_tokens
    return parsed


def score_extraction(extracted: dict, expected: dict) -> dict:
    metrics = {}
    for key in ["preferences", "goals", "expertise", "patterns"]:
        expected_items  = expected.get(key, [])
        extracted_items = extracted.get(key, [])
        extracted_texts = [
            (item.get("fact") or item.get("goal") or item.get("topic") or item.get("pattern") or "").lower()
            for item in extracted_items
        ]
        expected_count  = len(expected_items)
        extracted_count = len(extracted_items)
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


def check_hallucinations(extracted: dict, conversation_text: str) -> list[str]:
    conv_lower = conversation_text.lower()
    hallucinations = []
    for key in ["preferences", "goals", "expertise", "patterns"]:
        for item in extracted.get(key, []):
            item_text = (item.get("fact") or item.get("goal") or
                         item.get("topic") or item.get("pattern") or "")
            sig_words = [w for w in item_text.lower().split() if len(w) > 5]
            if sig_words:
                def word_in_conv(w):
                    if w in conv_lower: return True
                    return w[:5] in conv_lower
                miss_rate = sum(1 for w in sig_words if not word_in_conv(w)) / len(sig_words)
                if miss_rate > 0.5:
                    hallucinations.append(item_text)
    return hallucinations


def run_experiment():
    client = anthropic.Anthropic(api_key=API_KEY)

    print(f"\n{'='*60}")
    print("Experiment 15c — Option D: Claude API Extraction")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")

    all_results   = []
    all_recalls   = []
    total_in_tok  = 0
    total_out_tok = 0

    for conv in CONVERSATIONS:
        print(f"\nConversation {conv['id']}: {conv['description']}")
        conversation_text = format_conversation(conv["turns"])
        extracted = extract_with_claude(conversation_text, client)

        parse_ok    = "_parse_failed" not in extracted
        latency     = extracted.pop("_latency_ms", None)
        raw         = extracted.pop("_raw", "")
        in_tok      = extracted.pop("_input_tokens", 0)
        out_tok     = extracted.pop("_output_tokens", 0)
        total_in_tok  += in_tok
        total_out_tok += out_tok

        metrics = score_extraction(extracted, conv["expected"])
        hallucinations = check_hallucinations(extracted, conversation_text)

        print(f"  Parse success: {parse_ok} | Latency: {latency}ms | Tokens: {in_tok}in/{out_tok}out")
        for key, m in metrics.items():
            if m["expected"] > 0:
                print(f"  {key:12s}: expected={m['expected']} | extracted={m['extracted']} | "
                      f"found={m['found']} | recall={m['recall']:.0%}")
        if hallucinations:
            print(f"  Possible hallucinations: {hallucinations}")
        else:
            print("  No hallucinations detected.")

        for key, m in metrics.items():
            if m["recall"] is not None:
                all_recalls.append(m["recall"])

        all_results.append({
            "conversation_id": conv["id"],
            "description":     conv["description"],
            "parse_success":   parse_ok,
            "latency_ms":      latency,
            "input_tokens":    in_tok,
            "output_tokens":   out_tok,
            "extracted":       extracted,
            "expected":        conv["expected"],
            "metrics":         metrics,
            "hallucinations":  hallucinations,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")

    parse_rate = sum(1 for r in all_results if r["parse_success"]) / len(all_results)
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    total_hall = sum(len(r["hallucinations"]) for r in all_results)
    est_cost   = (total_in_tok / 1_000_000 * 0.25) + (total_out_tok / 1_000_000 * 1.25)

    print(f"Parse rate:       {parse_rate:.0%}  (baseline: 100%)")
    print(f"Average recall:   {avg_recall:.0%}  (baseline: 69%)")
    print(f"Hallucinations:   {total_hall}      (baseline: 1)")
    print(f"Total tokens:     {total_in_tok}in / {total_out_tok}out")
    print(f"Estimated cost:   ${est_cost:.4f} for {len(CONVERSATIONS)} conversations")
    print(f"Cost per session: ~${est_cost/len(CONVERSATIONS):.4f}")

    if avg_recall >= 0.80 and total_hall == 0:
        decision = "PASS — Claude API extraction is production-ready"
    elif avg_recall > 0.69:
        decision = "PASS — Claude API beats local baseline; use as extraction fallback"
    else:
        decision = "PARTIAL — marginal improvement; consider prompt tuning"

    print(f"\nDecision: {decision}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":       "exp_15c_option_d_claude_api",
        "model":            MODEL,
        "run_at":           ts,
        "parse_rate":       parse_rate,
        "avg_recall":       avg_recall,
        "total_hallucinations": total_hall,
        "total_input_tokens":   total_in_tok,
        "total_output_tokens":  total_out_tok,
        "estimated_cost_usd":   est_cost,
        "decision":         decision,
        "conversations":    all_results,
    }
    json_path = RESULTS_DIR / f"run_optionD_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
