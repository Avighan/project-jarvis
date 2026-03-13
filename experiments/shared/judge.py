"""
LLM-as-Judge for Project Jarvis experiments.
Uses local Ollama to rate/compare AI responses without human input.
"""

import re
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model


COMPARE_PROMPT = """\
You are evaluating two AI assistant responses.

User context memories injected:
{memories}

User's question:
{prompt}

--- Response A ---
{resp_a}

--- Response B ---
{resp_b}
---

Rate each response 1-5 where:
5 = Excellent: directly uses the user context, concise, expert-level
4 = Good: mostly relevant, minor issues
3 = Average: somewhat useful but generic
2 = Poor: largely ignores user context
1 = Bad: irrelevant or harmful

The user is a 10-year Python expert who prefers direct answers without preamble.
Prefer responses that actually use the injected memories.

Reply with ONLY this exact format (two lines, nothing else):
A: <number>
B: <number>"""


RATE_PROMPT = """\
Rate this AI assistant response 1-5.

User's question: {prompt}
Injected memories: {memories}

Response:
{response}

Rating criteria (5=best):
5 = Uses injected context, concise, expert-appropriate
4 = Good, minor flaws
3 = Average, somewhat generic
2 = Mostly ignores context
1 = Poor

The user is a 10-year Python expert who dislikes verbose answers.
Reply with ONLY a single integer 1-5, nothing else."""


def compare(prompt: str, resp_a: str, resp_b: str,
            memories: list[str] | None = None,
            model: str | None = None) -> tuple[int, int]:
    """
    Ask LLM to rate response A and B.
    Returns (rating_a, rating_b) as ints 1-5.
    Defaults to (3, 3) on parse failure.
    """
    model = model or pick_model()
    mem_str = "\n".join(f"- {m}" for m in (memories or [])) or "(none)"
    full = COMPARE_PROMPT.format(
        memories=mem_str,
        prompt=prompt,
        resp_a=resp_a[:600],
        resp_b=resp_b[:600],
    )
    raw, _ = generate(full, model=model, temperature=0.1, max_tokens=20)
    return _parse_ab(raw)


def rate(prompt: str, response: str,
         memories: list[str] | None = None,
         model: str | None = None) -> int:
    """
    Ask LLM to rate a single response 1-5.
    Returns int, defaults to 3 on parse failure.
    """
    model = model or pick_model()
    mem_str = "\n".join(f"- {m}" for m in (memories or [])) or "(none)"
    full = RATE_PROMPT.format(
        prompt=prompt,
        memories=mem_str,
        response=response[:600],
    )
    raw, _ = generate(full, model=model, temperature=0.1, max_tokens=10)
    nums = re.findall(r"[1-5]", raw.strip())
    return int(nums[0]) if nums else 3


def _parse_ab(raw: str) -> tuple[int, int]:
    """Parse 'A: 4\nB: 3' style output."""
    a_match = re.search(r"A\s*:\s*([1-5])", raw, re.IGNORECASE)
    b_match = re.search(r"B\s*:\s*([1-5])", raw, re.IGNORECASE)
    ra = int(a_match.group(1)) if a_match else 3
    rb = int(b_match.group(1)) if b_match else 3
    return ra, rb
