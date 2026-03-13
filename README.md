# Project Jarvis — PoC

Everything here runs on your local machine only. No cloud calls. No API keys.

## Quick Start

```bash
# 1. Initialise the database
python3 POC/core/setup_db.py

# 2. Ask Jarvis something
python3 POC/core/jarvis_cli.py "How should I structure this Python module?"

# 3. Ask without memory (baseline)
python3 POC/core/jarvis_cli.py --no-memory "same question"

# 4. View stored memories
python3 POC/core/jarvis_cli.py --show-memories

# 5. Rate a response
python3 POC/core/jarvis_cli.py --rate <interaction_id> 4
```

## Prerequisites

- Ollama running: `ollama serve`
- Models pulled: `llama3:latest` (already done)
- Python 3.12+ with `requests` and `numpy` (already installed)

## Experiment Sequence

Run experiments in order — each one sets a parameter used by the next:

```bash
python3 POC/experiments/experiment_01/run.py   # Gate: does injection work?
python3 POC/experiments/experiment_02/run.py   # Sets injection format
python3 POC/experiments/experiment_03/run.py   # Sets top_N
python3 POC/experiments/experiment_04/run.py   # Sets extraction prompt
python3 POC/experiments/experiment_05/run.py   # Sets retrieval method
python3 POC/experiments/experiment_06/run.py   # Sets disinhibition viability
pip install networkx
python3 POC/experiments/experiment_07/run.py   # Sets graph vs flat decision
python3 POC/experiments/experiment_08/run.py   # Sets confidence weighting
```

After running experiments, update these values in `POC/core/jarvis_cli.py`:
- `TOP_N` (from Experiment 3)
- `INJECTION_FORMAT` (from Experiment 2)
- `CONFIDENCE_WEIGHT` (from Experiment 8)

## Structure

```
POC/
  core/                   Core PoC implementation (runnable today)
    setup_db.py           One-time DB init
    memory_store.py       Read/write for all 3 memory tiers
    retrieval.py          TF-IDF + cosine + disinhibition retrieval
    ollama_client.py      Local Ollama API wrapper
    extractor.py          Session-end fact extraction
    working_memory.py     Memory injection formatter
    jarvis_cli.py         Main CLI

  experiments/
    shared/               Shared helpers (db, ollama)
    experiment_01/ - 08/  One folder per experiment; run.py + results/

  2026-03-12-poc-exploratory.pdf   Full experiment guide
  README.md                        This file
```
