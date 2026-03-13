# POC-Jarvis_v1

Two interfaces over the same `core/` memory engine. Both isolated, both share the same SQLite store at `~/.jarvis/jarvis_poc.db`.

## Structure

```
POC-Jarvis_v1/
  web-app/        Option A — FastAPI + HTML frontend (localhost:8000)
  desktop-app/    Option B — Electron native Mac app
```

## Shared Core

Both apps import from `../core/` — same memory_store, retrieval, extractor, working_memory, ollama_client.
No duplication. One DB. One source of truth.

## Run Web App

```bash
cd web-app
pip install fastapi uvicorn
python main.py
# Open http://localhost:8000
```

## Run Desktop App

```bash
cd desktop-app
npm install
npm start
```
