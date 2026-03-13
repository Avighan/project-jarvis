"""
Project Jarvis PoC — Working memory builder.
Assembles the context block injected into every Ollama call.
Injection format is set by experiment_02; default is 'structured'.
"""

import json
from typing import Literal

InjectionFormat = Literal["json", "prose", "structured"]


def format_memories(
    memories: list[dict],
    fmt: InjectionFormat = "structured",
) -> str:
    """
    Format a list of retrieved memories into a context block for injection.

    Three formats tested in Experiment 2:
      json       — machine-readable JSON object
      prose      — natural language paragraph
      structured — tagged lines (recommended default after Experiment 2)
    """
    if not memories:
        return ""

    if fmt == "json":
        payload = {
            "user_context": [
                {
                    "fact": m["content"],
                    "category": m.get("category", "general"),
                    "confidence": round(m.get("confidence", 0.7), 2),
                }
                for m in memories
            ]
        }
        return f"<user_context>\n{json.dumps(payload, indent=2)}\n</user_context>\n\n"

    if fmt == "prose":
        lines = ["Context about this user:"]
        for m in memories:
            conf = m.get("confidence", 0.7)
            conf_label = "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low"
            lines.append(f"- {m['content']} (confidence: {conf_label})")
        return "\n".join(lines) + "\n\n"

    # Default: structured tag format
    lines = []
    for m in memories:
        conf = m.get("confidence", 0.7)
        conf_label = "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low"
        lines.append(
            f"[MEMORY: {m['content']} | category: {m.get('category','general')} | confidence: {conf_label}]"
        )
    return "\n".join(lines) + "\n\n"


def build_prompt(
    user_input: str,
    memories: list[dict],
    fmt: InjectionFormat = "structured",
    system_note: str = "",
) -> str:
    """
    Assemble the full prompt to send to Ollama.
    Structure:
      [memory context block]
      [optional system note]
      User: {user_input}
    """
    parts = []

    memory_block = format_memories(memories, fmt)
    if memory_block:
        parts.append(memory_block)

    if system_note:
        parts.append(f"Note: {system_note}\n\n")

    parts.append(f"User: {user_input}")
    return "".join(parts)


def build_prompt_no_memory(user_input: str) -> str:
    """Baseline prompt with no memory context. Used in Experiment 1 Test A."""
    return f"User: {user_input}"
