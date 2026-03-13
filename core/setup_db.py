"""
Project Jarvis PoC — One-time database initialisation.
Run once: python3 POC/core/setup_db.py
Creates: ~/.jarvis/jarvis_poc.db
"""

import sqlite3
import os
import pathlib

DB_DIR  = pathlib.Path.home() / ".jarvis"
DB_PATH = DB_DIR / "jarvis_poc.db"


def init_db(db_path: pathlib.Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    conn.commit()
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    -- ── Tier 1: Episodic log ──────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS interactions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT    NOT NULL DEFAULT (datetime('now')),
        session_id       TEXT    NOT NULL,
        user_input       TEXT    NOT NULL,
        jarvis_response  TEXT    NOT NULL,
        model_used       TEXT    NOT NULL DEFAULT 'llama3:latest',
        latency_ms       INTEGER,
        user_rating      INTEGER,  -- 1-5, NULL if not rated
        user_edit        TEXT,     -- user's correction if they rewrote the response
        memories_injected TEXT,    -- JSON: list of memory_ids injected this turn
        injection_format TEXT      -- 'json' | 'prose' | 'structured'
    );

    -- ── Tier 2A: Semantic memories (flat store, PoC) ──────────────────────────
    CREATE TABLE IF NOT EXISTS memories (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        content          TEXT    NOT NULL,
        category         TEXT    NOT NULL DEFAULT 'general',
                                          -- preference|expertise|goal|pattern|general
        confidence       REAL    NOT NULL DEFAULT 0.7,  -- 0.0-1.0
        created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
        last_confirmed_at TEXT,
        source_session   TEXT,            -- session_id that generated this
        access_count     INTEGER NOT NULL DEFAULT 0,
        embedding        BLOB              -- nomic-embed-text vector (added in Exp 5)
    );

    -- ── Tier 2B: Structured fact store ───────────────────────────────────────
    CREATE TABLE IF NOT EXISTS profile (
        key              TEXT PRIMARY KEY,
        value            TEXT NOT NULL,
        updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS expertise (
        topic            TEXT PRIMARY KEY,
        level            TEXT NOT NULL DEFAULT 'intermediate',
                                          -- novice|intermediate|expert
        evidence_count   INTEGER NOT NULL DEFAULT 1,
        trajectory       TEXT    NOT NULL DEFAULT 'stable',
                                          -- growing|stable|rusty
        last_updated     TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS preferences (
        category         TEXT NOT NULL,
        preference       TEXT NOT NULL,
        strength         INTEGER NOT NULL DEFAULT 3,  -- 1-5
        last_confirmed   TEXT    NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (category, preference)
    );

    CREATE TABLE IF NOT EXISTS goals (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        goal             TEXT    NOT NULL,
        priority         INTEGER NOT NULL DEFAULT 3,  -- 1-5
        status           TEXT    NOT NULL DEFAULT 'active',
        deadline         TEXT,
        created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    -- ── Memory graph edges (for Experiment 7) ────────────────────────────────
    CREATE TABLE IF NOT EXISTS memory_edges (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id        INTEGER NOT NULL REFERENCES memories(id),
        target_id        INTEGER NOT NULL REFERENCES memories(id),
        edge_type        TEXT    NOT NULL,
                                          -- contradicts|supports|follows_from|
                                          -- causes|weakened_by|depends_on
        weight           REAL    NOT NULL DEFAULT 1.0,
        created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    -- ── Experiment results log ────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS experiment_runs (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment       TEXT    NOT NULL,  -- e.g. 'exp_01_memory_injection'
        run_at           TEXT    NOT NULL DEFAULT (datetime('now')),
        result_json      TEXT    NOT NULL,  -- full structured results
        summary          TEXT              -- human-readable 1-paragraph summary
    );
    """)


def seed_demo_memories(conn: sqlite3.Connection) -> None:
    """Insert 5 demo facts for quick testing. Safe to re-run (skips if exists)."""
    demo_facts = [
        ("user prefers direct responses without preamble or filler phrases",
         "preference", 0.9),
        ("user is building Project Jarvis — a persistent AI memory layer in Python 3.12",
         "goal", 0.95),
        ("user has 10 years of software engineering experience; expert-level Python",
         "expertise", 0.9),
        ("user is running Ollama locally on MacBook Pro M3 Pro with 36GB unified memory",
         "general", 0.85),
        ("user dislikes over-explained answers — wants the answer, not the lecture",
         "preference", 0.9),
    ]
    existing = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if existing == 0:
        conn.executemany(
            "INSERT INTO memories (content, category, confidence) VALUES (?,?,?)",
            demo_facts
        )
        conn.commit()
        print(f"Seeded {len(demo_facts)} demo memories.")
    else:
        print(f"Memories table already has {existing} rows — skipping seed.")


if __name__ == "__main__":
    conn = init_db()
    print(f"Database initialised at: {DB_PATH}")
    seed_demo_memories(conn)
    conn.close()
    print("Done. Run: python3 POC/core/jarvis_cli.py \"your question\"")
