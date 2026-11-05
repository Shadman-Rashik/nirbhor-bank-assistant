import json
import threading
import urllib.error
import urllib.request

import pytest

from nirbhor.config import Settings
from nirbhor.pipeline import Assistant
from nirbhor.retriever import FaqRetriever
from nirbhor.webapp import WebApp, make_server


@pytest.fixture()
def ui(faq, tmp_path):
    bot = Assistant(faq, FaqRetriever(faq, None), None, Settings(use_embeddings=False))   # retrieval-only: no model needed
    server = make_server(WebApp(bot, "test-model", tmp_path), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()


def call(url, data=None, ctype="application/json", raw=None):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype} if body is not None else {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_page_and_static_files_are_served_with_security_headers(ui):
    base, _ = ui
    for path, ctype in (("/", "text/html"), ("/static/style.css", "text/css"), ("/static/app.js", "text/javascript")):
        code, headers, body = call(base + path)
        assert code == 200 and headers["Content-Type"].startswith(ctype) and body
        assert "default-src 'self'" in headers["Content-Security-Policy"] and headers["X-Content-Type-Options"] == "nosniff"
    page = call(base + "/")[2].decode()
    assert "http://" not in page and "https://" not in page          # loads nothing from the internet


def test_nothing_else_can_be_read_from_disk(ui):
    base, _ = ui
    for path in ("/static/../README.md", "/README.md", "/static/%2e%2e/README.md", "/src/nirbhor/config.py", "/data/bank-faq.csv"):
        assert call(base + path)[0] == 404


def test_ask_returns_an_answer_and_a_handover_for_unknown_questions(ui):
    base, _ = ui
    code, _, body = call(base + "/api/ask", {"question": "What is the late payment fee on a credit card?"})
    a = json.loads(body)
    assert code == 200 and a["status"] == "answered" and a["faq_ids"] == [23] and a["lang"] == "en" and "500" in a["text"]
    b = json.loads(call(base + "/api/ask", {"question": "আজকের স্বর্ণের দাম কত?"})[2])
    assert b["status"] == "idk" and b["lang"] == "bn" and b["faq_ids"] == []


def test_bad_requests_are_refused_politely(ui):
    base, _ = ui
    assert call(base + "/api/ask", ctype="text/plain", raw=b'{"question":"hi"}')[0] == 415      # another site cannot post here
    assert call(base + "/api/ask", raw=b"not json")[0] == 400
    assert call(base + "/api/ask", {"question": 5})[0] == 400
    assert call(base + "/api/ask", raw=json.dumps({"question": "x" * 20000}).encode())[0] == 413
    assert call(base + "/api/nope", {"question": "x"})[0] == 404
    assert json.loads(call(base + "/api/ask", {"question": "   "})[2])["reason"] == "empty_input"


def test_handover_is_written_to_the_queue_and_health_works(ui):
    base, logs = ui
    assert call(base + "/api/handover", {"question": "gold price?", "lang": "en", "reason": "retrieval_below_threshold"})[0] == 200
    row = json.loads((logs / "handover_requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["question"] == "gold price?" and row["lang"] == "en"
    assert json.loads(call(base + "/api/health")[2]) == {"ready": True, "model": "test-model"}
