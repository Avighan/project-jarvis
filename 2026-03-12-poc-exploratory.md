---
title: "Project Jarvis — PoC Exploratory Areas"
subtitle: "Local-Only Experiments · What to Validate · How to Run It"
date: "March 12, 2026"
---

# Project Jarvis — PoC Exploratory Areas

**Scope:** Everything in this document runs on your local machine only.
No Anthropic API. No Vast.ai. No external services.
Models used: `llama3:latest` (already pulled), `zephyr:latest` (fallback).
Embeddings (Experiment 5 only): `nomic-embed-text` — one `ollama pull nomic-embed-text`.

---

## What the PoC Is Proving

One thing: **persistent memory makes Jarvis meaningfully better than a stateless LLM.**

Everything else in the architecture — the proactive engine, causal graph, fine-tuning, data
connectors — is an enhancement on this foundation. If the foundation doesn't hold, nothing
else matters. The 8 experiments below validate that foundation from different angles.

---

## What Runs Locally Right Now (Zero Extra Installs)

| Component | Implementation | Deps |
| --- | --- | --- |
| Episodic log | SQLite (`sqlite3` built-in) | None |
| Structured fact store | SQLite | None |
| Implicit extraction | Ollama API (`requests`) | None |
| Working memory builder | Pure Python string assembly | None |
| TF-IDF retrieval | `numpy` (already installed) | None |
| Task type classifier | Ollama API | None |
| CLI interface | Pure Python | None |

**One `pip install` each:**

- `pip install networkx` — memory graph (Experiment 7)
- `pip install mcp` — MCP server for Claude Desktop (Phase 1 deliverable)

**One `ollama pull` for Experiment 5 only:**

- `ollama pull nomic-embed-text` (~274MB) — local vector embeddings

---

## Folder Structure

```text
POC/
  core/
    memory_store.py      SQLite episodic log + fact store
    retrieval.py         TF-IDF keyword retrieval (numpy only)
    ollama_client.py     Ollama API wrapper (localhost:11434)
    extractor.py         Session-end implicit fact extraction
    working_memory.py    Working memory builder + injection formatter
    jarvis_cli.py        Main CLI entry point
    setup_db.py          One-time database initialisation

  experiments/
    shared/
      ollama.py          Shared Ollama call helper
      db.py              Shared SQLite helper
    experiment_01/       Memory injection: does it matter at all?
    experiment_02/       Injection format: JSON vs prose vs structured
    experiment_03/       Memory count: how many before quality drops?
    experiment_04/       Implicit extraction quality
    experiment_05/       Retrieval: TF-IDF keyword vs vector embeddings
    experiment_06/       Disinhibition: task classification accuracy
    experiment_07/       Memory graph vs flat list
    experiment_08/       Confidence decay: help or noise?

  2026-03-12-poc-exploratory.pdf   This document
```

---

## The Absolute Minimum Viable Local PoC

Five days. Zero cloud calls. Works on `llama3:latest` already pulled.

```text
Day 1:  setup_db.py runs — creates SQLite schema
        memory_store.py — write path working
        ollama_client.py — llama3 responding via requests

Day 2:  extractor.py — session-end extraction prompt
        working_memory.py — top-4 memories assembled as context block

Day 3:  jarvis_cli.py — python jarvis_cli.py "your question"
        Full loop: query → retrieve → inject → Ollama → log → extract

Day 4:  Run Experiment 1 — does memory injection improve responses?
        If yes: core premise validated. Ship MCP server.

Day 5:  pip install mcp → MCP server wrapper
        Jarvis accessible from Claude Desktop, all local
```

---

## Exploratory Experiments

---

### Experiment 1 — Does Memory Injection Matter at All?

**The foundational question. Run this before anything else.**

**Setup:**

Store 5 "past facts" in SQLite:

```text
Fact 1: User prefers direct responses without preamble
Fact 2: User is building Project Jarvis — a personal AI memory layer
Fact 3: User has 10 years of software engineering experience
Fact 4: User is working with Python 3.12 and Ollama locally
Fact 5: User dislikes over-explained answers — wants the answer, not the lecture
```

Write 10 test prompts that implicitly require these facts to answer well:

```text
Prompt 1:  "How should I structure this Python module?"
Prompt 2:  "What's the best way to think about this architecture decision?"
Prompt 3:  "Give me a quick take on this design."
... (see run.py for all 10)
```

**Test:**

```text
Test A:  Prompt → Ollama (no memory injected)
Test B:  Prompt → Ollama (top-4 memories prepended as context)
```

**Measure:**

- Does Test B reference past context (e.g., uses "direct" style, skips preamble)?
- Blind-rate A vs B responses on relevance (1–5)

**Expected:** B wins clearly — responses are shorter, more targeted, skip scaffolding.
**If not:** The injection format is wrong, or the model is ignoring the context block.
This is the gate that determines whether to proceed.

**Run:** `python3 POC/experiments/experiment_01/run.py`

---

### Experiment 2 — Which Injection Format Does llama3 Respond Best To?

Same 10 memories. Same 10 prompts. Three injection formats.

**Format A — JSON:**

```json
{
  "user_context": [
    {"fact": "user prefers direct responses", "confidence": 0.9},
    {"fact": "user is building Project Jarvis", "confidence": 0.95}
  ]
}
```

**Format B — Prose paragraph:**

```text
Context about this user: They prefer direct responses without preamble.
They are an experienced software engineer (10 years) building Project Jarvis,
a personal AI memory system, in Python 3.12 on a local Ollama setup.
```

**Format C — Structured tags:**

```text
[MEMORY: user prefers direct responses — no preamble (confidence: high)]
[MEMORY: user is building Project Jarvis in Python 3.12 (confidence: high)]
[MEMORY: 10 years software engineering experience (confidence: high)]
```

**Measure:**

- Which format produces responses that actually reflect the injected context?
- Which format causes the model to explicitly reference or ignore the memories?
- Latency difference between formats (minor, but worth noting)

**This is a 2-hour experiment. The result directly determines the working memory template
used in `working_memory.py` for all subsequent development.**

**Run:** `python3 POC/experiments/experiment_02/run.py`

---

### Experiment 3 — How Many Memories Before Quality Drops?

**Setup:** Pre-seed 32 memories of varying relevance. 5 test prompts.

**Test:** Inject N = 2, 4, 8, 16, 32 memories per prompt.

**Measure:**

```text
(a) Response latency — time.time() around the Ollama call
(b) Does the model get confused? (contradictory statements, hallucinations)
(c) Does it actually USE more memories, or just ignore the extras?
    → Check: count how many injected memories appear to influence the response
(d) Response quality (1–5 blind rating)
```

**Context window reality check:**

```text
llama3:latest (8B):       ~8K context. At ~50 tokens/memory:
                          N=32 = ~1,600 tokens just for context.
                          Leaves ~6,400 for prompt + response. Tight but feasible.

llama3.3:latest (30B):    128K context. Memory injection not a concern.
                          Pull with: ollama pull llama3.3:latest (~19GB)
```

**Expected:** Quality peaks at N=4–8 for llama3 8B, degrades around N=16+.
**Sets:** The `top_N` parameter in `working_memory.py`.

**Run:** `python3 POC/experiments/experiment_03/run.py`

---

### Experiment 4 — Implicit Extraction Quality

**The session-end extraction loop. Can llama3 reliably extract structured facts?**

**Setup:** 5 pre-written conversation excerpts (hardcoded in run.py):

```text
Excerpt 1: Technical discussion where user reveals Python preference, project context
Excerpt 2: Planning discussion revealing deadline, priority, frustration with a tool
Excerpt 3: Personal work-style discussion revealing communication preferences
Excerpt 4: Research conversation revealing domain expertise and knowledge gaps
Excerpt 5: Ambiguous conversation with minimal extractable facts (edge case)
```

**Extraction prompt (tested in run.py):**

```text
"Extract all facts about the user from this conversation.
Return ONLY valid JSON with this exact schema:
{
  "preferences": [{"fact": string, "confidence": 0.0-1.0}],
  "expertise": [{"topic": string, "level": "novice|intermediate|expert", "confidence": 0.0-1.0}],
  "goals": [{"goal": string, "confidence": 0.0-1.0}],
  "patterns": [{"pattern": string, "confidence": 0.0-1.0}]
}
If nothing extractable, return empty arrays. Do not invent facts."
```

**Measure:**

```text
(a) What did it extract correctly? (compare to ground truth in run.py)
(b) What did it miss?
(c) Did it hallucinate facts not in the conversation?
(d) Was the JSON parseable? (json.loads() without exception)
(e) Schema consistency: did it follow the exact schema across all 5 excerpts?
```

**Expected finding:** llama3 8B extracts 70–80% of facts correctly but produces
inconsistent JSON schemas ~30% of the time. The fix: stricter schema in the prompt
or a post-processing pass. This experiment determines which.

**Run:** `python3 POC/experiments/experiment_04/run.py`

---

### Experiment 5 — Retrieval: TF-IDF Keyword vs. Vector Embeddings

**Do you need to pull `nomic-embed-text`, or is keyword matching sufficient for PoC scale?**

**Setup:** 50 pre-seeded memories in SQLite. 20 queries with known correct answers.

**Method A — TF-IDF (no extra installs):**

```python
# Uses only numpy (already installed)
# Compute term frequency over memory corpus
# Score each memory against query by overlapping terms (IDF-weighted)
# Return top-N by score
```

**Method B — Vector embeddings (requires `ollama pull nomic-embed-text`):**

```python
# Embed query and all memories via Ollama /api/embeddings endpoint
# Cosine similarity → top-N
```

**Measure:**

```text
Recall@1:  Did the top result match the expected memory?
Recall@5:  Did the expected memory appear in top 5 results?
Latency:   Time per query for each method (Method B is slower but more accurate)
```

**Expected:** At 50 memories, TF-IDF gets ~70% Recall@5, embeddings get ~85–90%.
**Decision gate:** If TF-IDF Recall@5 ≥ 75%, use it for PoC. Pull embeddings for Phase 2.

**Run:** `python3 POC/experiments/experiment_05/run.py`
Requires: `ollama pull nomic-embed-text` for Method B (Method A runs without it).

---

### Experiment 6 — Disinhibition: Does Task Classification Work?

**Can llama3 8B reliably classify queries by task type?**

**Setup:** 30 test prompts, 5 per task type (hardcoded in run.py):

```text
PLANNING:    "Help me structure my work for this week"
RESEARCH:    "What do I know about transformer attention mechanisms?"
EXECUTION:   "Create a Linear ticket for the auth bug"
EMOTIONAL:   "I'm feeling overwhelmed with everything on my plate"
REFLECTION:  "How have my work patterns changed over the last month?"
LEARNING:    "Explain how Granger causality works"
```

**Classification prompt:**

```text
"Classify this request into exactly one of these categories:
PLANNING | RESEARCH | EXECUTION | EMOTIONAL | REFLECTION | LEARNING

Request: {query}

Reply with only the category name. Nothing else."
```

**Measure:**

```text
Overall accuracy:    correct / 30
Per-class accuracy:  correct / 5 for each type
Confusion patterns:  which pairs does it confuse most?
```

**Decision gate:**
- ≥80% overall → disinhibition model is viable, proceed
- 65–79% → viable with a fallback rule for confused pairs
- <65% → task classification is unreliable on llama3 8B; use simpler retrieval for PoC

**Run:** `python3 POC/experiments/experiment_06/run.py`

---

### Experiment 7 — Memory Graph vs. Flat List

**The most uncertain experiment. Build the graph only if this shows value.**

**Setup (requires `pip install networkx`):**

```text
Flat list:    50 memories in SQLite, retrieved by relevance score
Graph:        Same 50 memories as networkx nodes
              Typed edges added between related memories:
                "user prefers brevity" --supports--> "user is an expert"
                "user finds X useful" --follows_from--> "user worked on Y"
```

**10 test prompts that require connecting two related memories:**

```text
Prompt 1: "How should I calibrate my explanation depth?"
  → Requires connecting: "user is expert" + "user prefers brevity"
  → Expected: skip scaffolding, go straight to answer

Prompt 2: "What's relevant background for my current project?"
  → Requires connecting: "working on Jarvis" + "memory architecture decisions"
...
```

**Measure:**

```text
Flat list: Does top-N retrieval surface BOTH connected memories for each prompt?
Graph:     Does edge traversal surface the pair that flat list misses?
```

**Honest caveat:** At 50 memories, the flat list may surface both memories naturally
via relevance scoring. The graph earns its complexity only when the two connected
memories are semantically distant but logically related (low cosine similarity, high
logical dependency). This experiment identifies whether that case exists at PoC scale.

**Decision:** If graph doesn't outperform flat list on >60% of prompts, delay graph to Phase 2.

**Run:** `pip install networkx && python3 POC/experiments/experiment_07/run.py`

---

### Experiment 8 — Confidence Decay: Does It Help or Add Noise?

**Should the PoC implement confidence-weighted retrieval, or is it premature?**

**Setup:** 100 pre-seeded memories with varying confidence:

```text
High confidence (0.8–1.0):  Recent, frequently confirmed facts
Medium confidence (0.5–0.7): Older or partially corroborated facts
Low confidence (0.2–0.4):   Contradicted or stale facts
```

**Test A — Relevance-only retrieval:**

```python
score = tfidf_similarity(query, memory)
top_N = sorted(memories, key=lambda m: score[m], reverse=True)[:N]
```

**Test B — Confidence-weighted retrieval:**

```python
score = tfidf_similarity(query, memory) * memory.confidence
top_N = sorted(memories, key=lambda m: score[m], reverse=True)[:N]
```

**5 test prompts with known high-confidence and low-confidence relevant memories.**

**Measure:**

```text
Which method surfaces the high-confidence memories more reliably?
Does down-weighting low-confidence memories improve response quality?
Does it accidentally exclude useful low-confidence memories?
```

**Expected:** At 100 memories, the effect is small but measurable. If Test B improves
response quality on ≥3/5 prompts, implement confidence weighting. If no difference, use
the schema (confidence field exists) but ignore it in the retrieval score for now.

**Run:** `python3 POC/experiments/experiment_08/run.py`

---

## What Each Experiment Decides

| Experiment | Decision it makes |
| --- | --- |
| 1 — Memory injection | Go / no-go on the entire PoC premise |
| 2 — Injection format | Sets the working memory template format |
| 3 — Memory count | Sets `top_N` in `working_memory.py` |
| 4 — Extraction quality | Sets extraction prompt strictness and post-processing need |
| 5 — Retrieval method | Sets whether to use TF-IDF or pull `nomic-embed-text` |
| 6 — Task classification | Sets whether disinhibition model is viable on llama3 8B |
| 7 — Graph vs flat list | Sets whether to build networkx graph now or delay to Phase 2 |
| 8 — Confidence decay | Sets whether confidence weighting is active in retrieval score |

---

## Running Order

Run in this sequence. Each experiment informs the next:

```text
1 → Gate: if no improvement, stop and re-examine premise
2 → Sets injection format for all subsequent experiments
3 → Sets top_N for all subsequent experiments
4 → Sets extraction prompt for the core system
5 → Sets retrieval method for experiments 6, 7, 8
6 → Sets whether disinhibition is in Phase 1 or Phase 2
7 → Sets whether graph is in Phase 1 or Phase 2
8 → Sets whether confidence weighting is active
```

---

## Result Recording

Each `run.py` saves results to its `results/` subfolder:

```text
experiment_0X/
  results/
    run_YYYYMMDD_HHMMSS.json    Full structured results
    run_YYYYMMDD_HHMMSS.txt     Human-readable summary
```

After completing all 8 experiments, update the architecture spec and PoC implementation
with the validated parameters. Do not assume any parameter value without running the
experiment that determines it.
