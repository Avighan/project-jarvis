"""
Experiment 12 — Graph-Augmented Retrieval at 100+ Memories
===========================================================
Question: Does graph-augmented retrieval beat flat TF-IDF at 100+ memories
          with manually seeded edges?

Method A: retrieve_tfidf(query, memories, top_n=4, confidence_weight=True)
Method B: TF-IDF top-2 + graph expansion via connected memories → merge → top 4

Pass: graph_wins >= 6/10 AND avg judge score for B >= avg judge score for A
Decision printed at end.

Run: python3 POC/experiments/experiment_12/run.py
"""

import sys, json, pathlib, datetime, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from experiments.shared.ollama import generate, pick_model
from experiments.shared.db     import fresh_db, seed_memories, get_memories
from experiments.shared.judge  import compare
from core.retrieval             import retrieve_tfidf
from core.working_memory        import build_prompt

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Core 20 memories (seeded in order, tracked as mem[0]..mem[19]) ─────────
CORE_MEMORIES: list[dict] = [
    {"content": "user prefers direct responses without preamble",           "category": "preference", "confidence": 0.92},  # 0
    {"content": "user is building Project Jarvis in Python 3.12",          "category": "goal",       "confidence": 0.95},  # 1
    {"content": "user has 10 years Python experience; expert level",        "category": "expertise",  "confidence": 0.92},  # 2
    {"content": "user uses MacBook Pro M3 Pro with 36GB RAM",               "category": "general",    "confidence": 0.95},  # 3
    {"content": "user wants Jarvis to work completely offline",             "category": "goal",       "confidence": 0.92},  # 4
    {"content": "user has Ollama running with llama3:latest locally",       "category": "general",    "confidence": 0.90},  # 5
    {"content": "user dislikes over-explained responses",                   "category": "preference", "confidence": 0.90},  # 6
    {"content": "user prefers minimal Python dependencies",                 "category": "preference", "confidence": 0.88},  # 7
    {"content": "user is testing PoC with SQLite and networkx",             "category": "general",    "confidence": 0.85},  # 8
    {"content": "user works in 2-hour morning focus blocks",                "category": "pattern",    "confidence": 0.85},  # 9
    {"content": "user tracks tasks in Linear",                              "category": "pattern",    "confidence": 0.82},  # 10
    {"content": "user prefers synchronous Python code for PoC",            "category": "preference", "confidence": 0.82},  # 11
    {"content": "user prefers Python type hints in all function signatures","category": "preference", "confidence": 0.80},  # 12
    {"content": "user finds context switching disruptive to deep work",     "category": "pattern",    "confidence": 0.80},  # 13
    {"content": "user uses GitHub for version control",                     "category": "general",    "confidence": 0.80},  # 14
    {"content": "user prefers writing tests after implementation in PoC",   "category": "preference", "confidence": 0.65},  # 15
    {"content": "user's secondary language is JavaScript",                  "category": "expertise",  "confidence": 0.50},  # 16
    {"content": "user has tried several AI tools before Jarvis",            "category": "general",    "confidence": 0.65},  # 17
    {"content": "user is interested in building a SaaS version of Jarvis",  "category": "goal",       "confidence": 0.60},  # 18
    {"content": "user has used BM25 in a previous Java search project",     "category": "expertise",  "confidence": 0.60},  # 19
]

# Low-conf async memory added before edge connection (will be index 20)
ASYNC_MEMORY = {"content": "user now prefers async Python", "category": "preference", "confidence": 0.20}

# ── Filler templates (cycle to pad to 100) ───────────────────────────────────
FILLER_TEMPLATES = [
    ("user mentioned liking jazz music",                               "pattern",    0.35),
    ("user has a dog named Max",                                       "general",    0.30),
    ("user grew up in London",                                         "general",    0.45),
    ("user drinks coffee not tea",                                     "pattern",    0.50),
    ("user has a standing desk setup",                                 "general",    0.55),
    ("user reads technical papers on weekends",                        "pattern",    0.65),
    ("user is interested in privacy-preserving AI",                    "preference", 0.75),
    ("user has experience with SQL databases",                         "expertise",  0.70),
    ("user uses a password manager",                                   "general",    0.50),
    ("user follows AI researchers on Twitter",                         "pattern",    0.60),
    ("user is aware of context window limitations of LLMs",            "expertise",  0.75),
    ("user has worked on distributed systems before",                  "expertise",  0.55),
    ("user prefers functional programming style sometimes",            "preference", 0.60),
    ("user mentioned wanting to write a technical blog post",          "goal",       0.55),
    ("user is aware of RAG architectures",                             "expertise",  0.70),
    ("user has used Redis for caching in a previous role",             "expertise",  0.50),
    ("user prefers environment variables for secrets management",       "preference", 0.70),
    ("user has experience with CI/CD pipelines",                       "expertise",  0.60),
    ("user tracks version history with git carefully",                 "preference", 0.75),
    ("user has used Jupyter notebooks for data exploration",           "expertise",  0.55),
]


def make_all_memories() -> list[dict]:
    """Build the full 101-memory list: 20 core + 1 async + 80 filler."""
    mems = list(CORE_MEMORIES) + [ASYNC_MEMORY]
    # Pad to 101 (20 core + 1 async + 80 filler) by cycling through templates
    filler_count = 0
    cycle_idx = 0
    while filler_count < 80:
        text, cat, conf = FILLER_TEMPLATES[cycle_idx % len(FILLER_TEMPLATES)]
        variant = filler_count // len(FILLER_TEMPLATES)
        content = text if variant == 0 else f"{text} (note {variant + 1})"
        mems.append({"content": content, "category": cat, "confidence": conf})
        filler_count += 1
        cycle_idx += 1
    return mems


# ── Graph helper: query connected memories using in-memory conn ───────────
def get_connected_in_memory(conn, memory_id: int) -> list[dict]:
    """Fetch memories connected to memory_id via memory_edges in the in-memory DB."""
    rows = conn.execute(
        """SELECT m.*, e.edge_type, e.weight
           FROM memories m
           JOIN memory_edges e ON (e.target_id = m.id OR e.source_id = m.id)
           WHERE (e.source_id = ? OR e.target_id = ?) AND m.id != ?""",
        (memory_id, memory_id, memory_id)
    ).fetchall()
    return [dict(r) for r in rows]


def add_edge(conn, source_id: int, target_id: int, edge_type: str, weight: float = 1.0) -> None:
    """Insert an edge directly into the in-memory connection."""
    valid = {"contradicts", "supports", "follows_from", "causes", "weakened_by", "depends_on"}
    assert edge_type in valid, f"Invalid edge_type: {edge_type}"
    conn.execute(
        "INSERT INTO memory_edges (source_id, target_id, edge_type, weight) VALUES (?,?,?,?)",
        (source_id, target_id, edge_type, weight)
    )
    conn.commit()


def retrieve_graph_augmented(
    query: str,
    conn,
    memories: list[dict],
    top_n: int = 4,
) -> list[dict]:
    """
    Method B: TF-IDF top-2 → get connected memories for each seed →
    merge + deduplicate by id → re-rank by confidence desc → top 4.
    """
    seeds = retrieve_tfidf(query, memories, top_n=2, confidence_weight=True)
    id_to_mem: dict[int, dict] = {m["id"]: m for m in memories}
    merged: dict[int, dict] = {}

    for seed in seeds:
        merged[seed["id"]] = seed
        connected = get_connected_in_memory(conn, seed["id"])
        for c in connected:
            cid = c["id"]
            if cid not in merged and cid in id_to_mem:
                merged[cid] = id_to_mem[cid]

    # Re-rank by confidence desc, take top 4
    ranked = sorted(merged.values(), key=lambda m: m.get("confidence", 0.0), reverse=True)
    return ranked[:top_n]


# ── 10 test queries ───────────────────────────────────────────────────────────
# Each entry: (query_text, target_memory_index_in_core, description)
TEST_QUERIES = [
    (
        "What response style does this user prefer?",
        [0, 6],
        "should find preamble pref (0) AND connected over-explanation pref (6)",
    ),
    (
        "Is this user likely to accept new dependencies for the PoC?",
        [7, 4, 11],
        "mem[7] minimal deps + connected offline goal (4) + sync pref (11)",
    ),
    (
        "What local tools is this user running for the project?",
        [5, 1],
        "mem[5] Ollama + connected Jarvis goal (1)",
    ),
    (
        "Does this user prefer sync or async Python?",
        [11, 20],
        "mem[11] sync pref vs contradicted async memory (index 20)",
    ),
    (
        "When is the best time to interrupt this user?",
        [9, 13],
        "mem[9] morning focus + connected context switching (13)",
    ),
    (
        "What is this user building and what stack are they using?",
        [1, 8, 5],
        "mem[1] Jarvis + connected SQLite/networkx (8) + Ollama (5)",
    ),
    (
        "What is this user's Python code style?",
        [11, 7],
        "mem[11] sync + connected minimal deps (7)",
    ),
    (
        "How much retrieval experience does this user have?",
        [19, 2],
        "mem[19] BM25 Java + connected Python expert (2)",
    ),
    (
        "What are this user's work pattern preferences?",
        [9, 13],
        "mem[9] morning blocks + mem[13] context switching",
    ),
    (
        "Should Jarvis use many small packages or few large ones?",
        [7, 4],
        "mem[7] minimal deps + connected offline goal (4)",
    ),
]


def run_experiment() -> None:
    model = pick_model()
    print(f"\n{'='*65}")
    print("Experiment 12 — Graph-Augmented Retrieval at 100+ Memories")
    print(f"Model: {model}")
    print(f"{'='*65}\n")

    # ── Seed memories ────────────────────────────────────────────────────────
    conn = fresh_db()
    all_mem_defs = make_all_memories()
    all_ids = seed_memories(conn, all_mem_defs)

    # Map positional index → DB id
    # Index 0-19: core memories, index 20: async memory, 21+: fillers
    mem_id = all_ids  # mem_id[i] = DB id for position i

    memories = get_memories(conn)
    print(f"Seeded {len(memories)} memories.")
    print(f"  Core 20: indices 0-19, IDs {mem_id[0]}–{mem_id[19]}")
    print(f"  Async memory: index 20, ID {mem_id[20]}")
    print(f"  Filler: indices 21-100\n")

    # ── Add edges ────────────────────────────────────────────────────────────
    print("Adding 20 typed edges...")

    # Primary semantic edges among core memories
    primary_edges = [
        (0,  6,  "supports"),       # no preamble supports no over-explanation
        (0,  12, "supports"),       # direct responses + type hints
        (1,  8,  "follows_from"),   # building Jarvis → uses SQLite/networkx
        (1,  5,  "follows_from"),   # building Jarvis → uses Ollama
        (4,  7,  "supports"),       # offline goal → minimal deps
        (4,  11, "supports"),       # offline goal → sync code
        (7,  11, "supports"),       # minimal deps supports sync preference
        (9,  13, "follows_from"),   # morning focus → context switching disruptive
        (2,  19, "supports"),       # Python expert + BM25 Java = retrieval experience
        (11, 20, "contradicts"),    # sync pref contradicts async memory (index 20)
    ]
    for src_idx, tgt_idx, etype in primary_edges:
        add_edge(conn, mem_id[src_idx], mem_id[tgt_idx], etype)
        print(f"  Edge: mem[{src_idx}] --{etype}--> mem[{tgt_idx}]")

    # 10 additional edges among filler memories (using positional index pairs)
    # Filler memories start at index 21 (after 20 core + 1 async)
    filler_pairs = [(0, 7), (1, 4), (2, 19), (3, 5), (6, 11), (9, 13), (10, 14), (7, 15), (8, 1), (12, 0)]
    for src_idx, tgt_idx in filler_pairs:
        # These use the core memory indices (0-19) not filler indices
        try:
            add_edge(conn, mem_id[src_idx], mem_id[tgt_idx], "supports")
        except Exception:
            pass  # Skip duplicate edges silently

    edge_count = conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
    print(f"Total edges in DB: {edge_count}\n")

    # ── Run 10 test queries ──────────────────────────────────────────────────
    all_results = []
    graph_wins = 0
    total_score_a = 0
    total_score_b = 0
    mem_contents = [d["content"] for d in all_mem_defs]

    for qi, (query, target_indices, description) in enumerate(TEST_QUERIES, 1):
        print(f"Query {qi:2d}/10: {query[:60]}...")
        print(f"         ({description})")

        # Target DB IDs (only indices 0-20 are valid)
        target_db_ids = set()
        for idx in target_indices:
            if idx < len(mem_id):
                target_db_ids.add(mem_id[idx])

        # Method A: flat TF-IDF top-4
        results_a = retrieve_tfidf(query, memories, top_n=4, confidence_weight=True)
        ids_a = {m["id"] for m in results_a}

        # Method B: graph-augmented
        results_b = retrieve_graph_augmented(query, conn, memories, top_n=4)
        ids_b = {m["id"] for m in results_b}

        # Did Method B retrieve a target memory that Method A missed?
        a_found_targets = target_db_ids & ids_a
        b_found_targets = target_db_ids & ids_b
        b_extra_targets = b_found_targets - a_found_targets
        graph_advantage = len(b_extra_targets) > 0

        if graph_advantage:
            graph_wins += 1

        # Generate LLM responses
        mem_str_a = [m["content"] for m in results_a]
        mem_str_b = [m["content"] for m in results_b]
        resp_a, lat_a = generate(build_prompt(query, results_a, fmt="structured"), model=model, max_tokens=300)
        resp_b, lat_b = generate(build_prompt(query, results_b, fmt="structured"), model=model, max_tokens=300)

        # Judge comparison
        score_a, score_b = compare(query, resp_a, resp_b, mem_str_a + mem_str_b, model=model)
        total_score_a += score_a
        total_score_b += score_b

        winner_label = "B" if score_b > score_a else ("A" if score_a > score_b else "tie")
        print(f"         Targets found — A: {len(a_found_targets)}/{len(target_db_ids)}"
              f"  B: {len(b_found_targets)}/{len(target_db_ids)}"
              f"  graph_advantage={graph_advantage}")
        print(f"         Judge scores — A:{score_a}  B:{score_b}  winner={winner_label}"
              f"  latency A:{lat_a}ms B:{lat_b}ms")

        all_results.append({
            "query_id":        qi,
            "query":           query,
            "description":     description,
            "target_ids":      list(target_db_ids),
            "a_found_targets": len(a_found_targets),
            "b_found_targets": len(b_found_targets),
            "graph_advantage": graph_advantage,
            "method_a": {
                "top_memories": [m["content"][:70] for m in results_a],
                "response":     resp_a,
                "latency_ms":   lat_a,
                "judge_score":  score_a,
            },
            "method_b": {
                "top_memories": [m["content"][:70] for m in results_b],
                "response":     resp_b,
                "latency_ms":   lat_b,
                "judge_score":  score_b,
            },
            "winner": winner_label,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    avg_a = total_score_a / len(TEST_QUERIES)
    avg_b = total_score_b / len(TEST_QUERIES)
    b_judge_wins = sum(1 for r in all_results if r["winner"] == "B")

    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"Graph advantage (B retrieved target A missed): {graph_wins}/10")
    print(f"Judge wins — B: {b_judge_wins}/10")
    print(f"Avg judge score — A: {avg_a:.2f}  B: {avg_b:.2f}")

    pass_retrieval = graph_wins >= 6
    pass_quality   = avg_b >= avg_a

    if pass_retrieval and pass_quality:
        decision = (
            f"PASS — graph augmentation works at 100 memories "
            f"(graph_wins={graph_wins}/10, avg_B={avg_b:.2f} >= avg_A={avg_a:.2f}); "
            f"build retrieve_graph() in retrieval.py"
        )
    elif pass_retrieval or pass_quality:
        decision = (
            f"PARTIAL — marginal improvement "
            f"(graph_wins={graph_wins}/10, avg_B={avg_b:.2f} vs avg_A={avg_a:.2f}); "
            f"revisit at 300+ memories"
        )
    else:
        decision = (
            f"FAIL/DEFER — flat TF-IDF still sufficient at this scale "
            f"(graph_wins={graph_wins}/10, avg_B={avg_b:.2f} vs avg_A={avg_a:.2f})"
        )

    print(f"\nDecision: {decision}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":    "exp_12_graph_augmented_retrieval",
        "model":         model,
        "run_at":        ts,
        "total_memories": len(memories),
        "total_edges":   edge_count,
        "graph_wins":    graph_wins,
        "b_judge_wins":  b_judge_wins,
        "avg_score_a":   round(avg_a, 3),
        "avg_score_b":   round(avg_b, 3),
        "decision":      decision,
        "results":       all_results,
    }
    json_path = RESULTS_DIR / f"run_{ts}.json"
    json_path.write_text(json.dumps(output, indent=2))
    txt_path  = RESULTS_DIR / f"run_{ts}.txt"
    txt_path.write_text(
        f"Experiment 12 — Graph-Augmented Retrieval at 100+ Memories\n"
        f"Run: {ts} | Model: {model}\n"
        f"Memories: {len(memories)} | Edges: {edge_count}\n"
        f"Graph wins (retrieval advantage): {graph_wins}/10\n"
        f"Judge wins B: {b_judge_wins}/10\n"
        f"Avg score A: {avg_a:.2f} | Avg score B: {avg_b:.2f}\n"
        f"Decision: {decision}\n"
    )
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    run_experiment()
