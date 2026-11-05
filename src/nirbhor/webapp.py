"""A small local web UI for the assistant. Python standard library only (no Streamlit, no Flask, no extra install).

  python scripts/serve_ui.py        -> open http://127.0.0.1:8000

Safety choices: listens on this computer only (127.0.0.1), serves only four fixed static files, accepts JSON only
(so another website cannot make your browser post to it), limits the request size, never shows error details to
the customer, and the page loads nothing from the internet.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
PAGES = {"/": ("index.html", "text/html; charset=utf-8")}
STATIC = {"/static/style.css": ("style.css", "text/css; charset=utf-8"),
          "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/static/logo.png": ("logo.png", "image/png")}
MAX_BODY = 8 * 1024
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


class WebApp:
    def __init__(self, assistant, model_label: str | None, log_dir: str | Path):
        self.assistant, self.model_label, self.log_dir = assistant, model_label, Path(log_dir)
        self._lock = threading.Lock()      # one question at a time: a small local model has one GPU

    def ask(self, question: str) -> dict:
        with self._lock:
            a = self.assistant.answer(question)
        return {"text": a.text, "status": a.status, "reason": a.reason, "lang": a.lang, "faq_ids": a.faq_ids,
                "ms": round(a.total_ms), "best_score": a.hits[0]["score"] if a.hits else None}

    def handover(self, question: str, lang: str, reason: str) -> None:
        """Demo hand-over queue: one line per request. A real deployment would feed the bank's staff queue instead."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question[:500], "lang": lang[:20], "reason": reason[:60]}
        with self._lock, open(self.log_dir / "handover_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_server(app: WebApp, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "BankAssistantUI"

        def log_message(self, fmt, *args):   # keep the console quiet: customer questions must not be echoed into logs
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in PAGES or path in STATIC:
                name, ctype = PAGES.get(path) or STATIC[path]
                return self._send(200, (WEB_DIR / name).read_bytes(), ctype)
            if path == "/api/health":
                return self._json(200, {"ready": True, "model": app.model_label})
            self._json(404, {"error": "not found"})

        def _body(self) -> dict | None:
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                self._json(415, {"error": "JSON only"})
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                self._json(400, {"error": "empty body"})
                return None
            if length > MAX_BODY:
                self._json(413, {"error": "too large"})
                return None
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError
                return data
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": "bad JSON"})
                return None

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path not in ("/api/ask", "/api/handover"):
                return self._json(404, {"error": "not found"})
            data = self._body()
            if data is None:
                return
            question = data.get("question")
            if not isinstance(question, str):
                return self._json(400, {"error": "question must be text"})
            if path == "/api/ask":
                return self._json(200, app.ask(question))
            app.handover(question, str(data.get("lang", "")), str(data.get("reason", "")))
            self._json(200, {"ok": True})

    return ThreadingHTTPServer((host, port), Handler)
