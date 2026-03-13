"""Shared DB helpers: spin up a fresh in-memory SQLite for each experiment."""
import sqlite3, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from core.setup_db import _create_tables


def fresh_db() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with full Jarvis schema.
    Each experiment gets its own isolated database — no cross-contamination."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def seed_memories(conn: sqlite3.Connection, facts: list[dict]) -> list[int]:
    """
    Seed a list of fact dicts into memories table.
    Each dict: {"content": str, "category": str, "confidence": float}
    Returns list of inserted IDs.
    """
    ids = []
    for f in facts:
        cur = conn.execute(
            "INSERT INTO memories (content, category, confidence) VALUES (?,?,?)",
            (f["content"], f.get("category", "general"), f.get("confidence", 0.7))
        )
        ids.append(cur.lastrowid)
    conn.commit()
    return ids


def get_memories(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM memories").fetchall()
    return [dict(r) for r in rows]
