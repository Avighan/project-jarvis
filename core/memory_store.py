"""
Project Jarvis PoC — Memory store: read/write operations for all three tiers.
"""

import json
import sqlite3
import pathlib
from typing import Optional

DB_PATH = pathlib.Path.home() / ".jarvis" / "jarvis_poc.db"


def get_conn(db_path: pathlib.Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run: python3 POC/core/setup_db.py"
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Tier 1: Episodic log ──────────────────────────────────────────────────────

def log_interaction(
    session_id: str,
    user_input: str,
    jarvis_response: str,
    model_used: str = "llama3:latest",
    latency_ms: Optional[int] = None,
    memories_injected: Optional[list[int]] = None,
    injection_format: Optional[str] = None,
    db_path: pathlib.Path = DB_PATH,
) -> int:
    """Write an interaction to the episodic log. Returns the row id."""
    conn = get_conn(db_path)
    cur = conn.execute(
        """INSERT INTO interactions
           (session_id, user_input, jarvis_response, model_used,
            latency_ms, memories_injected, injection_format)
           VALUES (?,?,?,?,?,?,?)""",
        (
            session_id, user_input, jarvis_response, model_used,
            latency_ms,
            json.dumps(memories_injected) if memories_injected else None,
            injection_format,
        )
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def rate_interaction(
    interaction_id: int,
    rating: int,
    edit: Optional[str] = None,
    db_path: pathlib.Path = DB_PATH,
) -> None:
    """Record a user rating (1-5) and optional correction for a logged interaction."""
    assert 1 <= rating <= 5, "Rating must be 1–5"
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE interactions SET user_rating=?, user_edit=? WHERE id=?",
        (rating, edit, interaction_id)
    )
    conn.commit()
    conn.close()


def recent_interactions(n: int = 10, db_path: pathlib.Path = DB_PATH) -> list[dict]:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Tier 2A: Semantic memories ────────────────────────────────────────────────

def add_memory(
    content: str,
    category: str = "general",
    confidence: float = 0.7,
    source_session: Optional[str] = None,
    db_path: pathlib.Path = DB_PATH,
) -> int:
    """Add a new memory. Returns memory id."""
    conn = get_conn(db_path)
    cur = conn.execute(
        """INSERT INTO memories (content, category, confidence, source_session)
           VALUES (?,?,?,?)""",
        (content, category, confidence, source_session)
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def all_memories(db_path: pathlib.Path = DB_PATH) -> list[dict]:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM memories ORDER BY confidence DESC, access_count DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_memory_accessed(memory_id: int, db_path: pathlib.Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """UPDATE memories
           SET access_count = access_count + 1,
               last_confirmed_at = datetime('now')
           WHERE id = ?""",
        (memory_id,)
    )
    conn.commit()
    conn.close()


def update_confidence(
    memory_id: int,
    delta: float,  # positive = corroboration, negative = contradiction
    db_path: pathlib.Path = DB_PATH,
) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """UPDATE memories
           SET confidence = MAX(0.0, MIN(1.0, confidence + ?))
           WHERE id = ?""",
        (delta, memory_id)
    )
    conn.commit()
    conn.close()


def add_memory_edge(
    source_id: int,
    target_id: int,
    edge_type: str,
    weight: float = 1.0,
    db_path: pathlib.Path = DB_PATH,
) -> None:
    """Add a typed edge between two memories (for Experiment 7 graph traversal)."""
    valid_types = {"contradicts", "supports", "follows_from", "causes", "weakened_by", "depends_on"}
    assert edge_type in valid_types, f"edge_type must be one of {valid_types}"
    conn = get_conn(db_path)
    conn.execute(
        "INSERT INTO memory_edges (source_id, target_id, edge_type, weight) VALUES (?,?,?,?)",
        (source_id, target_id, edge_type, weight)
    )
    conn.commit()
    conn.close()


def get_connected_memories(memory_id: int, db_path: pathlib.Path = DB_PATH) -> list[dict]:
    """Return all memories directly connected to the given memory via any edge."""
    conn = get_conn(db_path)
    rows = conn.execute(
        """SELECT m.*, e.edge_type, e.weight FROM memories m
           JOIN memory_edges e ON (e.target_id = m.id OR e.source_id = m.id)
           WHERE (e.source_id = ? OR e.target_id = ?) AND m.id != ?""",
        (memory_id, memory_id, memory_id)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Tier 2B: Structured fact store ────────────────────────────────────────────

def set_profile(key: str, value: str, db_path: pathlib.Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO profile (key, value, updated_at) VALUES (?,?,datetime('now'))",
        (key, value)
    )
    conn.commit()
    conn.close()


def get_profile(db_path: pathlib.Path = DB_PATH) -> dict[str, str]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT key, value FROM profile").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def upsert_expertise(
    topic: str,
    level: str,
    trajectory: str = "stable",
    db_path: pathlib.Path = DB_PATH,
) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO expertise (topic, level, trajectory) VALUES (?,?,?)
           ON CONFLICT(topic) DO UPDATE SET
             level=excluded.level,
             evidence_count=evidence_count+1,
             trajectory=excluded.trajectory,
             last_updated=datetime('now')""",
        (topic, level, trajectory)
    )
    conn.commit()
    conn.close()


def upsert_preference(
    category: str,
    preference: str,
    strength_delta: int = 0,
    db_path: pathlib.Path = DB_PATH,
) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO preferences (category, preference, strength) VALUES (?,?,3)
           ON CONFLICT(category, preference) DO UPDATE SET
             strength=MAX(1, MIN(5, strength + ?)),
             last_confirmed=datetime('now')""",
        (category, preference, strength_delta)
    )
    conn.commit()
    conn.close()


def add_goal(
    goal: str,
    priority: int = 3,
    deadline: Optional[str] = None,
    db_path: pathlib.Path = DB_PATH,
) -> int:
    conn = get_conn(db_path)
    cur = conn.execute(
        "INSERT INTO goals (goal, priority, deadline) VALUES (?,?,?)",
        (goal, priority, deadline)
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def log_experiment(
    experiment: str,
    result: dict,
    summary: str = "",
    db_path: pathlib.Path = DB_PATH,
) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "INSERT INTO experiment_runs (experiment, result_json, summary) VALUES (?,?,?)",
        (experiment, json.dumps(result), summary)
    )
    conn.commit()
    conn.close()
