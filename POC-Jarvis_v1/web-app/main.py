"""
Jarvis Web App — FastAPI backend
Wraps core/ modules. Runs on localhost:8000.
Supports both Ollama (local) and Claude Haiku (API) for chat inference.
"""

import sys, pathlib, uuid, sqlite3, time, os
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

import core.memory_store as ms
from core.retrieval      import retrieve_tfidf
from core.working_memory import format_memories
from core.extractor      import EXTRACTION_SCHEMA, EXTRACTION_PROMPT_TEMPLATE, _parse_json_response
from core.ollama_client  import generate, pick_model
from core.setup_db       import init_db

app = FastAPI(title="Jarvis", docs_url=None, redoc_url=None)

# ── Rate limiting (per IP) ─────────────────────────────────────────────────────
RATE_LIMIT_HOURLY = int(os.environ.get("RATE_LIMIT_HOURLY", 20))   # max /ask calls per hour
RATE_LIMIT_DAILY  = int(os.environ.get("RATE_LIMIT_DAILY",  100))  # max /ask calls per day

_rate_store: dict = defaultdict(lambda: {"hour": [], "day": []})

def _check_rate_limit(ip: str):
    now = time.time()
    bucket = _rate_store[ip]
    bucket["hour"] = [t for t in bucket["hour"] if now - t < 3600]
    bucket["day"]  = [t for t in bucket["day"]  if now - t < 86400]
    if len(bucket["hour"]) >= RATE_LIMIT_HOURLY:
        raise HTTPException(
            status_code=429,
            detail=f"Hourly limit reached ({RATE_LIMIT_HOURLY} messages/hour). Try again later."
        )
    if len(bucket["day"]) >= RATE_LIMIT_DAILY:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({RATE_LIMIT_DAILY} messages/day). Come back tomorrow."
        )
    bucket["hour"].append(now)
    bucket["day"].append(now)

def _get_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else request.client.host

_default_db = pathlib.Path(os.environ.get("DB_PATH", str(pathlib.Path.home() / ".jarvis" / "jarvis_poc.db")))
DB_PATH              = _default_db
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_EXTRACT_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_CHAT_MODEL    = "claude-haiku-4-5-20251001"

# Ensure DB is initialised (creates tables if first run)
init_db(DB_PATH)


def _claude_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _extract_with_claude(conversation: str) -> dict:
    client = _claude_client()
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        schema=EXTRACTION_SCHEMA,
        conversation=conversation.strip()
    )
    msg = client.messages.create(
        model=CLAUDE_EXTRACT_MODEL, max_tokens=800, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_response(msg.content[0].text)


def _generate_with_claude_chat(prompt: str) -> tuple:
    client = _claude_client()
    t0 = time.time()
    msg = client.messages.create(
        model=CLAUDE_CHAT_MODEL,
        max_tokens=1024,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.time() - t0) * 1000)
    return msg.content[0].text, latency_ms


# ── Request models ────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    inject_memory: bool = True
    model_preference: str = "auto"   # "auto" | "claude" | "ollama"

class AddMemoryRequest(BaseModel):
    content: str
    category: str = "preference"
    confidence: float = 0.9

class ExtractRequest(BaseModel):
    conversation: str

class RateRequest(BaseModel):
    interaction_id: int
    rating: int  # 1-5


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((pathlib.Path(__file__).parent / "templates" / "index.html").read_text())


@app.get("/limits")
async def limits(request: Request):
    ip  = _get_ip(request)
    now = time.time()
    bucket = _rate_store[ip]
    used_hour = len([t for t in bucket["hour"] if now - t < 3600])
    used_day  = len([t for t in bucket["day"]  if now - t < 86400])
    return {
        "hourly":  {"used": used_hour, "limit": RATE_LIMIT_HOURLY, "remaining": max(0, RATE_LIMIT_HOURLY - used_hour)},
        "daily":   {"used": used_day,  "limit": RATE_LIMIT_DAILY,  "remaining": max(0, RATE_LIMIT_DAILY  - used_day)},
    }


@app.get("/config")
async def config():
    """Return available models and backend capabilities."""
    ollama_available = False
    try:
        m = pick_model()
        ollama_available = bool(m)
    except Exception:
        pass
    return {
        "ollama_available": ollama_available,
        "claude_available": True,
        "default_model": "auto",
    }


@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    _check_rate_limit(_get_ip(request))
    context_block = ""
    memories_used = []
    hits = []

    if req.inject_memory:
        all_mems = ms.all_memories(DB_PATH)
        for m in all_mems:
            m.pop("embedding", None)
        if all_mems:
            hits = retrieve_tfidf(req.query, all_mems, top_n=4, confidence_weight=True)
            if hits:
                context_block = format_memories(hits, fmt="json")
                memories_used = [{"content": h["content"], "score": round(h.get("retrieval_score", 0), 3)} for h in hits]
                for h in hits:
                    ms.mark_memory_accessed(h["id"], DB_PATH)

    prompt = f"{context_block}User: {req.query}" if context_block else req.query

    # Route to Claude or Ollama based on preference
    pref = req.model_preference
    model_used = ""

    if pref == "claude":
        response, latency_ms = _generate_with_claude_chat(prompt)
        model_used = CLAUDE_CHAT_MODEL
    elif pref == "ollama":
        ollama_model = pick_model()
        response, latency_ms = generate(prompt, model=ollama_model, temperature=0.7)
        model_used = ollama_model
    else:
        # auto: try Ollama first, fall back to Claude
        try:
            ollama_model = pick_model()
            response, latency_ms = generate(prompt, model=ollama_model, temperature=0.7)
            model_used = ollama_model
        except Exception:
            response, latency_ms = _generate_with_claude_chat(prompt)
            model_used = CLAUDE_CHAT_MODEL

    session_id = str(uuid.uuid4())[:8]
    interaction_id = ms.log_interaction(
        session_id=session_id,
        user_input=req.query,
        jarvis_response=response,
        model_used=model_used,
        latency_ms=latency_ms,
        memories_injected=[h["id"] for h in hits] if hits else None,
        injection_format="json" if context_block else None,
        db_path=DB_PATH,
    )

    return {
        "response":       response,
        "memories_used":  memories_used,
        "interaction_id": interaction_id,
        "latency_ms":     latency_ms,
        "model":          model_used,
    }


@app.get("/memories")
async def list_memories(limit: int = 100):
    mems = ms.all_memories(DB_PATH)
    return {"memories": mems[:limit], "total": len(mems)}


@app.post("/memories/add")
async def add_memory(req: AddMemoryRequest):
    memory_id = ms.add_memory(req.content, req.category, req.confidence, db_path=DB_PATH)
    return {"memory_id": memory_id, "status": "added"}


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


@app.post("/extract")
async def extract(req: ExtractRequest, request: Request):
    _check_rate_limit(_get_ip(request))
    """Extract facts from pasted conversation. Does not auto-save."""
    result = _extract_with_claude(req.conversation)
    result.pop("_raw", None)
    result.pop("_parse_failed", None)
    return {"extracted": result}


@app.post("/extract-and-save")
async def extract_and_save(req: ExtractRequest):
    """Extract facts via Claude Haiku and immediately save them."""
    result = _extract_with_claude(req.conversation)
    result.pop("_raw", None)
    result.pop("_parse_failed", None)
    saved = 0
    for category, items in result.items():
        if not isinstance(items, list):
            continue
        for item in items:
            fact = item.get("fact") or item.get("goal") or item.get("topic") or item.get("pattern") or ""
            conf = float(item.get("confidence", 0.7))
            if fact:
                ms.add_memory(content=fact, category=category, confidence=conf, db_path=DB_PATH)
                saved += 1
    return {"extracted": result, "saved": saved}


@app.post("/rate")
async def rate(req: RateRequest):
    if not (1 <= req.rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be 1-5")
    ms.rate_interaction(req.interaction_id, req.rating, db_path=DB_PATH)
    return {"status": "rated"}


@app.get("/profile")
async def profile():
    return {"profile": ms.get_profile(DB_PATH)}


@app.get("/stats")
async def stats():
    mems   = ms.all_memories(DB_PATH)
    recent = ms.recent_interactions(10, DB_PATH)
    rated  = [r for r in recent if r.get("user_rating")]
    avg    = sum(r["user_rating"] for r in rated) / len(rated) if rated else None
    return {
        "total_memories":      len(mems),
        "recent_interactions": len(recent),
        "avg_rating":          round(avg, 2) if avg else None,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=not os.environ.get("PORT"))
