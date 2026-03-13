"""
Jarvis Desktop — pywebview launcher
Opens a native macOS WebKit window pointing at the FastAPI backend.
Spawns the backend if it's not already running.

Usage:
    python3 jarvis_window.py
"""

import sys
import pathlib
import subprocess
import time
import http.client

BACKEND_PORT = 8000
BACKEND_URL  = f"http://127.0.0.1:{BACKEND_PORT}"

THIS_DIR    = pathlib.Path(__file__).parent
WEB_APP_DIR = THIS_DIR.parent / "web-app"
PROJECT_ROOT = THIS_DIR.parent.parent.parent   # so `import core` works


def backend_up() -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=1)
        conn.request("GET", "/")
        conn.getresponse()
        return True
    except Exception:
        return False


def start_backend():
    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=WEB_APP_DIR,
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait up to 10 s for the backend to be ready
    for _ in range(20):
        time.sleep(0.5)
        if backend_up():
            print("[jarvis] backend ready")
            return proc
    print("[jarvis] warning: backend didn't respond in time — opening anyway")
    return proc


def main():
    proc = None
    if backend_up():
        print("[jarvis] backend already running")
    else:
        print("[jarvis] starting backend…")
        proc = start_backend()

    import webview
    window = webview.create_window(
        "Jarvis",
        BACKEND_URL,
        width=1200,
        height=800,
        min_size=(800, 600),
    )
    webview.start()

    if proc:
        proc.terminate()


if __name__ == "__main__":
    main()
