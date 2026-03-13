"""
Project Jarvis PoC — Session-end implicit fact extraction.
Calls local Ollama to extract structured facts from a conversation.
Tested and validated in Experiment 4.
"""

import json
import re
from typing import Optional
from .ollama_client import generate, pick_model

EXTRACTION_SCHEMA = """{
  "preferences": [{"fact": "<string>", "confidence": <0.0-1.0>}],
  "expertise":   [{"topic": "<string>", "level": "novice|intermediate|expert", "confidence": <0.0-1.0>}],
  "goals":       [{"goal": "<string>", "confidence": <0.0-1.0>}],
  "patterns":    [{"pattern": "<string>", "confidence": <0.0-1.0>}]
}"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract standing facts about the user from this conversation.
Standing facts = things persistently true about who the user is (preferences, skills, goals, habits).
Do NOT extract momentary actions, questions asked, or things the assistant said.

Return ONLY valid JSON with this exact schema:
{schema}

Rules:
- Only extract facts that are explicitly stated or very clearly implied by the user's own words.
- Do NOT infer traits from single actions. "User asked what time it is" → extract nothing.
- Do NOT paraphrase beyond the user's meaning. Stay close to what was said.
- Do NOT extract assistant statements as user facts.
- If a category has nothing extractable, return an empty array — do not invent items to fill it.
- confidence: 0.9=user stated it directly, 0.5=clearly implied, 0.2=weak single signal.

Example — what NOT to do (momentary question, not a standing fact):
  User says: "What time is it in London?"
  WRONG: extract "user wants to know current time"
  RIGHT: extract nothing

Example — what TO do (stated preference/pattern):
  User says: "I prefer short answers, no preamble."
  RIGHT: extract {{"fact": "prefers short answers without preamble", "confidence": 0.9}}

Example — what TO do (stated work pattern):
  User says: "I need at least 3 hours of uninterrupted time to make progress."
  RIGHT: extract pattern {{"pattern": "needs 3+ uninterrupted hours for deep work", "confidence": 0.9}}

Conversation:
{conversation}

JSON output:"""


def extract_facts(
    conversation: str,
    model: Optional[str] = None,
) -> dict:
    """
    Extract structured facts from a conversation string.
    Returns parsed dict, or empty structure on parse failure.
    """
    model = model or pick_model()
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        schema=EXTRACTION_SCHEMA,
        conversation=conversation.strip()
    )
    raw, latency_ms = generate(prompt, model=model, temperature=0.1, max_tokens=800)

    parsed = _parse_json_response(raw)
    parsed["_raw"] = raw
    parsed["_latency_ms"] = latency_ms
    parsed["_parse_success"] = "_raw" not in parsed or parsed.get("preferences") is not None
    return parsed


def _parse_json_response(raw: str) -> dict:
    """
    Try to parse JSON from model response.
    Models often wrap JSON in markdown code blocks or add prose preamble.
    """
    empty = {"preferences": [], "expertise": [], "goals": [], "patterns": []}

    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find first { to last }
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    # Return empty on all failures
    print(f"[extractor] Failed to parse JSON. Raw response:\n{raw[:300]}")
    return {**empty, "_parse_failed": True}


def format_conversation(turns: list[dict]) -> str:
    """
    Format a list of conversation turns for extraction.
    Each turn: {"role": "user"|"assistant", "content": str}
    """
    lines = []
    for turn in turns:
        role    = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
