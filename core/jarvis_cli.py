"""
Project Jarvis PoC — CLI entry point.
Usage: python3 POC/core/jarvis_cli.py "your question"
       python3 POC/core/jarvis_cli.py --session my_session "your question"
       python3 POC/core/jarvis_cli.py --rate <interaction_id> <1-5>
       python3 POC/core/jarvis_cli.py --show-memories
       python3 POC/core/jarvis_cli.py --add-memory "fact" [--category preference] [--confidence 0.9]
       python3 POC/core/jarvis_cli.py --extract "conversation text"

All calls go to local Ollama except --extract which uses Claude Haiku API.
"""

import sys
import uuid
import json
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.setup_db    import init_db, DB_PATH
from core.memory_store import (
    log_interaction, rate_interaction, all_memories,
    add_memory, mark_memory_accessed, get_profile,
)
from core.retrieval       import retrieve_tfidf
from core.working_memory  import build_prompt, build_prompt_no_memory, InjectionFormat
from core.ollama_client   import generate, pick_model

ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_EXTRACT_MODEL = "claude-haiku-4-5-20251001"

# Config (set by experiments)
TOP_N:            int             = 4
INJECTION_FORMAT: InjectionFormat = "json"
CONFIDENCE_WEIGHT: bool           = True


def extract_with_claude(conversation: str) -> dict:
    """Extract facts from conversation using Claude Haiku (Option D)."""
    try:
        import anthropic
        from core.extractor import EXTRACTION_SCHEMA, EXTRACTION_PROMPT_TEMPLATE, _parse_json_response
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            schema=EXTRACTION_SCHEMA,
            conversation=conversation.strip()
        )
        msg = client.messages.create(
            model=CLAUDE_EXTRACT_MODEL,
            max_tokens=800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json_response(msg.content[0].text)
    except ImportError:
        print("[error] anthropic not installed. Run: pip install anthropic")
        return {}
    except Exception as e:
        print(f"[error] Claude extraction failed: {e}")
        return {}


def ask(user_input: str, session_id: str, model: str, no_memory: bool = False) -> dict:
    memories = []
    injected_ids = []

    if not no_memory:
        raw_memories = all_memories()
        retrieved    = retrieve_tfidf(user_input, raw_memories, top_n=TOP_N, confidence_weight=CONFIDENCE_WEIGHT)
        memories     = retrieved
        injected_ids = [m["id"] for m in retrieved]
        for mid in injected_ids:
            mark_memory_accessed(mid)

    prompt = (
        build_prompt_no_memory(user_input) if no_memory
        else build_prompt(user_input, memories, fmt=INJECTION_FORMAT)
    )

    response, latency_ms = generate(prompt, model=model)

    interaction_id = log_interaction(
        session_id=session_id, user_input=user_input, jarvis_response=response,
        model_used=model, latency_ms=latency_ms,
        memories_injected=injected_ids if not no_memory else [],
        injection_format=INJECTION_FORMAT if not no_memory else None,
    )

    return {"interaction_id": interaction_id, "response": response,
            "latency_ms": latency_ms, "memories_used": len(injected_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Jarvis — local AI with persistent memory")
    parser.add_argument("query",         nargs="?", help="Your question or request")
    parser.add_argument("--session",     default=None)
    parser.add_argument("--no-memory",   action="store_true")
    parser.add_argument("--model",       default=None)
    parser.add_argument("--rate",        nargs=2, metavar=("ID", "RATING"))
    parser.add_argument("--show-memories", action="store_true")
    parser.add_argument("--add-memory",  metavar="CONTENT")
    parser.add_argument("--category",    default="preference",
                        help="preference|goal|expertise|pattern|general")
    parser.add_argument("--confidence",  type=float, default=0.9)
    parser.add_argument("--extract",     metavar="CONVERSATION",
                        help="Extract + save facts from conversation text using Claude Haiku")
    parser.add_argument("--extract-only", action="store_true",
                        help="With --extract: print facts but do NOT save")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("First run — initialising database...")
        init_db()

    if args.rate:
        rate_interaction(int(args.rate[0]), int(args.rate[1]))
        print(f"Rated interaction {args.rate[0]}: {args.rate[1]}/5")
        return

    if args.show_memories:
        memories = all_memories()
        if not memories:
            print("No memories stored yet.")
            return
        print(f"\n{len(memories)} memories:\n")
        for m in memories:
            print(f"  [{m['id']:3d}] [{m['category']:12s}] (conf:{m.get('confidence',0):.2f}) {m['content']}")
        return

    if args.add_memory:
        mem_id = add_memory(content=args.add_memory, category=args.category, confidence=args.confidence)
        print(f"Memory saved [id={mem_id}] [{args.category}] \"{args.add_memory}\"")
        return

    if args.extract:
        print(f"Extracting via Claude Haiku...")
        extracted = extract_with_claude(args.extract)
        if not extracted:
            print("Nothing extracted.")
            return
        saved = 0
        for category, items in extracted.items():
            if not isinstance(items, list):
                continue
            for item in items:
                fact = item.get("fact") or item.get("goal") or item.get("topic") or item.get("pattern") or ""
                conf = float(item.get("confidence", 0.7))
                if fact:
                    level = item.get("level", "")
                    print(f"  [{category}] (conf:{conf:.2f}) {fact}" + (f" [{level}]" if level else ""))
                    if not args.extract_only:
                        add_memory(content=fact, category=category, confidence=conf)
                        saved += 1
        print(f"\n{'(dry run)' if args.extract_only else f'{saved} facts saved.'}")
        return

    if not args.query:
        parser.print_help()
        return

    model      = args.model or pick_model()
    session_id = args.session or f"session_{uuid.uuid4().hex[:8]}"
    result     = ask(user_input=args.query, session_id=session_id, model=model, no_memory=args.no_memory)

    print("\n" + "─" * 60)
    print(result["response"])
    print("─" * 60)
    print(f"[{result['latency_ms']}ms | memories used: {result['memories_used']} | id: {result['interaction_id']}]")
    print(f"Rate: python3 POC/core/jarvis_cli.py --rate {result['interaction_id']} <1-5>")


if __name__ == "__main__":
    main()
