"""
Experiment 7 — Memory Graph vs. Flat List
==========================================
Does networkx graph traversal surface connected insights that
flat TF-IDF retrieval misses?

Tests 10 prompts requiring two related but semantically distant memories.
Decision: if graph wins on >6/10 prompts, build graph now; otherwise defer to Phase 2.

Requires: pip install networkx
Run: python3 POC/experiments/experiment_07/run.py
"""

import sys, json, pathlib, datetime, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

try:
    import networkx as nx
except ImportError:
    print("networkx not installed. Run: pip install networkx")
    sys.exit(1)

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from core.retrieval             import retrieve_tfidf
from core.working_memory        import build_prompt

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Memories that have logical connections (some semantically distant) ────────
MEMORIES = [
    # Group A: Expertise + preference pair (semantically distant, logically connected)
    {"id_hint": "A1", "content": "user has 10 years of Python engineering experience — expert level",
     "category": "expertise", "confidence": 0.9},
    {"id_hint": "A2", "content": "user prefers concise direct answers without scaffolding",
     "category": "preference", "confidence": 0.9},
    # Group B: Goal + constraint pair
    {"id_hint": "B1", "content": "user wants Jarvis to work offline with no cloud calls",
     "category": "goal", "confidence": 0.95},
    {"id_hint": "B2", "content": "user's machine has Ollama with llama3, zephyr, codegemma models",
     "category": "general", "confidence": 0.9},
    # Group C: Pattern + preference pair
    {"id_hint": "C1", "content": "user works in 2-hour deep focus blocks in the morning",
     "category": "pattern", "confidence": 0.8},
    {"id_hint": "C2", "content": "user dislikes interruptions during focused work sessions",
     "category": "preference", "confidence": 0.85},
    # Group D: Project + technology pair
    {"id_hint": "D1", "content": "user is building a SQLite-backed memory store for Jarvis",
     "category": "goal", "confidence": 0.9},
    {"id_hint": "D2", "content": "user prefers using WAL mode in SQLite for concurrent writes",
     "category": "preference", "confidence": 0.8},
    # Group E: Expertise + goal pair
    {"id_hint": "E1", "content": "user is learning causal inference and DoWhy library",
     "category": "expertise", "confidence": 0.7},
    {"id_hint": "E2", "content": "user wants to add a personal causal knowledge graph in Phase 3",
     "category": "goal", "confidence": 0.75},
    # Padding — unrelated memories to make retrieval harder
    {"id_hint": "P1", "content": "user prefers hiking in autumn",
     "category": "pattern", "confidence": 0.5},
    {"id_hint": "P2", "content": "user drinks coffee in the mornings",
     "category": "pattern", "confidence": 0.45},
    {"id_hint": "P3", "content": "user has tried many productivity apps",
     "category": "general", "confidence": 0.5},
    {"id_hint": "P4", "content": "user once mentioned reading a book on stoicism",
     "category": "pattern", "confidence": 0.4},
    {"id_hint": "P5", "content": "user's favourite film genre is sci-fi",
     "category": "pattern", "confidence": 0.4},
]

# Graph edges: which memories are logically connected
EDGES = [
    ("A1", "A2", "supports"),    # expertise supports preference for conciseness
    ("B1", "B2", "depends_on"),  # offline goal depends on local models
    ("C1", "C2", "follows_from"),# deep focus pattern implies dislike of interruptions
    ("D1", "D2", "depends_on"),  # SQLite store depends on WAL mode preference
    ("E1", "E2", "follows_from"),# learning causal inference leads to building causal graph
]

# 10 test prompts requiring connected memory pairs
CONNECTED_PROMPTS = [
    {
        "prompt": "How should I calibrate the depth and length of my answers to this user?",
        "needs": ("A1", "A2"),
        "insight": "Expert user + prefers conciseness → skip scaffolding, be direct",
    },
    {
        "prompt": "What local AI inference options does the user have for running Jarvis?",
        "needs": ("B1", "B2"),
        "insight": "Wants offline → uses Ollama with specific models already pulled",
    },
    {
        "prompt": "Should I send a notification to this user right now?",
        "needs": ("C1", "C2"),
        "insight": "Morning focus blocks + dislikes interruptions → check time before alerting",
    },
    {
        "prompt": "What SQLite configuration should the memory store use?",
        "needs": ("D1", "D2"),
        "insight": "Building SQLite memory store + prefers WAL mode → use WAL for concurrent writes",
    },
    {
        "prompt": "What should the user tackle next on their AI roadmap?",
        "needs": ("E1", "E2"),
        "insight": "Learning causal inference + plans causal graph in Phase 3 → connected progression",
    },
    {
        "prompt": "Is this user likely to be working right now at 9am?",
        "needs": ("C1", "C2"),
        "insight": "Morning focus blocks → yes, likely in deep work",
    },
    {
        "prompt": "How should Jarvis respond to a beginner-level question from this user?",
        "needs": ("A1", "A2"),
        "insight": "Expert level + hates scaffolding → still be direct even on basic questions",
    },
    {
        "prompt": "What database transaction approach should I use in the memory write path?",
        "needs": ("D1", "D2"),
        "insight": "SQLite store + WAL mode preference → use conn.execute with WAL and commit",
    },
    {
        "prompt": "Can Jarvis answer questions about Granger causality for this user?",
        "needs": ("E1", "E2"),
        "insight": "User is learning it + plans to use it → can engage at intermediate level",
    },
    {
        "prompt": "What response format should Jarvis use for a Python code question?",
        "needs": ("A1", "A2"),
        "insight": "Expert + no scaffolding → code directly, no explanation unless asked",
    },
]


def build_graph(memories: list[dict], id_hint_to_db_id: dict) -> nx.DiGraph:
    """Build a networkx DiGraph from memories and pre-defined edges."""
    G = nx.DiGraph()
    for m in memories:
        G.add_node(m["id"], content=m["content"], category=m["category"], confidence=m["confidence"])

    for hint_a, hint_b, edge_type in EDGES:
        db_a = id_hint_to_db_id.get(hint_a)
        db_b = id_hint_to_db_id.get(hint_b)
        if db_a and db_b:
            G.add_edge(db_a, db_b, type=edge_type)
            G.add_edge(db_b, db_a, type=edge_type)  # bidirectional for retrieval

    return G


def retrieve_graph(
    query: str,
    memories: list[dict],
    graph: nx.DiGraph,
    top_n: int = 4,
) -> list[dict]:
    """
    Graph retrieval: TF-IDF to find seed nodes, then expand via edges.
    Returns up to top_n memories including connected neighbours.
    """
    seed = retrieve_tfidf(query, memories, top_n=2)
    seed_ids = {m["id"] for m in seed}

    # Expand by following edges from seed nodes
    expanded_ids = set(seed_ids)
    for m in seed:
        if m["id"] in graph:
            neighbours = list(graph.neighbors(m["id"]))
            expanded_ids.update(neighbours)

    # Return all expanded memories, sorted by retrieval score
    id_to_mem = {m["id"]: m for m in memories}
    expanded_mems = [id_to_mem[mid] for mid in expanded_ids if mid in id_to_mem]
    return retrieve_tfidf(query, expanded_mems, top_n=top_n)


def run_experiment():
    model = pick_model()
    print(f"\n{'='*60}")
    print("Experiment 7 — Memory Graph vs. Flat List")
    print(f"Model: {model} | networkx version: {nx.__version__}")
    print(f"{'='*60}\n")

    conn = fresh_db()
    db_ids = seed_memories(conn, MEMORIES)
    memories = get_memories(conn)

    # Map id_hint to actual DB id
    id_hint_to_db_id = {}
    for i, m_def in enumerate(MEMORIES):
        id_hint_to_db_id[m_def["id_hint"]] = db_ids[i]

    graph = build_graph(memories, id_hint_to_db_id)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Print edges
    for hint_a, hint_b, edge_type in EDGES:
        da = id_hint_to_db_id.get(hint_a)
        db = id_hint_to_db_id.get(hint_b)
        m_a = next((m for m in MEMORIES if m["id_hint"] == hint_a), {})
        m_b = next((m for m in MEMORIES if m["id_hint"] == hint_b), {})
        print(f"  Edge: [{hint_a}] → [{hint_b}] ({edge_type})")
        print(f"    '{m_a['content'][:50]}' → '{m_b['content'][:50]}'")

    all_results = []
    graph_wins = 0

    for i, test in enumerate(CONNECTED_PROMPTS, 1):
        prompt   = test["prompt"]
        needs    = test["needs"]
        needed_db_ids = {id_hint_to_db_id[h] for h in needs if h in id_hint_to_db_id}

        # Method A — flat TF-IDF
        flat_results = retrieve_tfidf(prompt, memories, top_n=4)
        flat_ids     = {m["id"] for m in flat_results}
        flat_found   = needed_db_ids.issubset(flat_ids)

        # Method B — graph retrieval
        graph_results = retrieve_graph(prompt, memories, graph, top_n=4)
        graph_ids     = {m["id"] for m in graph_results}
        graph_found   = needed_db_ids.issubset(graph_ids)

        winner = "graph" if (graph_found and not flat_found) else \
                 "flat"  if (flat_found and not graph_found) else \
                 "tie"   if (flat_found and graph_found) else "neither"

        if winner == "graph":
            graph_wins += 1

        result = {
            "prompt_id":   i,
            "prompt":      prompt,
            "needs":       list(needs),
            "insight":     test["insight"],
            "flat_found":  flat_found,
            "graph_found": graph_found,
            "winner":      winner,
            "flat_retrieved":  [m["content"][:60] for m in flat_results],
            "graph_retrieved": [m["content"][:60] for m in graph_results],
        }
        all_results.append(result)

        f_sym = "✓" if flat_found else "✗"
        g_sym = "✓" if graph_found else "✗"
        print(f"\n  Prompt {i}: {prompt[:55]}...")
        print(f"    Flat={f_sym} Graph={g_sym} Winner={winner}")
        if not flat_found and not graph_found:
            print(f"    Neither found connected pair {needs}")

    # ── Summary ──────────────────────────────────────────────────────────────
    flat_wins   = sum(1 for r in all_results if r["winner"] == "flat")
    ties        = sum(1 for r in all_results if r["winner"] == "tie")
    neither     = sum(1 for r in all_results if r["winner"] == "neither")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Graph wins:  {graph_wins}/10")
    print(f"Flat wins:   {flat_wins}/10")
    print(f"Ties:        {ties}/10")
    print(f"Neither:     {neither}/10")

    if graph_wins > 6:
        decision = f"PASS — graph outperforms flat on {graph_wins}/10 prompts. Build networkx graph in Phase 1."
    elif graph_wins >= 4:
        decision = f"PARTIAL — graph wins {graph_wins}/10. Add graph but treat as enhancement, not core."
    else:
        decision = f"DEFER — graph wins only {graph_wins}/10 at this scale. Use flat list for PoC. Revisit at 500+ memories."

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":  "exp_07_graph_vs_flat",
        "model":       model,
        "run_at":      ts,
        "graph_wins":  graph_wins,
        "flat_wins":   flat_wins,
        "decision":    decision,
        "results":     all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 7 — Graph vs Flat\n"
        f"Run: {ts} | Model: {model}\n"
        f"Graph wins: {graph_wins}/10 | Flat wins: {flat_wins}/10\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
