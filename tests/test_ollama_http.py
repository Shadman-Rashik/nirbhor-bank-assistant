"""Tests our HTTP code (request format, caching, outputs, scripts) against a FAKE Ollama server on localhost.

This proves OUR code does what we intend. It cannot prove that the real Ollama behaves exactly like this fake
(the fake follows my understanding of /api/embed and /api/chat), so the first run on a real machine is still the real test.
"""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from nirbhor.config import Settings
from nirbhor.factory import build_assistant
from nirbhor.llm import OllamaChat
from nirbhor.retriever import OllamaEmbedder

ROOT = Path(__file__).resolve().parents[1]


def _vec(text, dim=64):
    v = [0.0] * dim
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v


class FakeOllama(BaseHTTPRequestHandler):
    calls: dict = {}

    def log_message(self, *args):
        pass

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send({"models": [{"name": "fake:latest"}, {"name": "bge-m3:latest"}]})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            FakeOllama.calls["embed"] += 1
            FakeOllama.calls["last_embed"] = body
            return self._send({"embeddings": [_vec(t) for t in body["input"]]})
        if self.path == "/api/chat":
            FakeOllama.calls["chat"] += 1
            FakeOllama.calls["last_chat"] = body
            system, user = body["messages"][0]["content"], body["messages"][-1]["content"]
            if "You check whether a bank FAQ entry" in system:
                FakeOllama.calls["judge"] += 1
                FakeOllama.calls["judge_examples"] += "Worked examples" in system
                text = "YES"
            else:
                m = re.search(r"\nA: (.*?)\n\nCustomer question", user, re.S)
                text = m.group(1) if m else "Hello"   # echoes the FAQ answer, like a perfectly obedient model
            return self._send({"message": {"role": "assistant", "content": text}, "prompt_eval_count": 11,
                               "eval_count": 7, "done": True})
        self.send_response(404)
        self.end_headers()


@pytest.fixture()
def fake_ollama():
    FakeOllama.calls = {"embed": 0, "chat": 0, "judge": 0, "judge_examples": 0, "last_embed": None, "last_chat": None}
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_embedder_batches_normalises_and_caches(fake_ollama, tmp_path):
    cache = tmp_path / "cache.json"
    e = OllamaEmbedder(fake_ollama, "bge-m3", cache_path=cache)
    m = e.embed(["loan interest", "branch hours"])
    assert m.shape == (2, 64) and np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert FakeOllama.calls["embed"] == 1 and FakeOllama.calls["last_embed"]["model"] == "bge-m3"
    e.embed(["loan interest"])                                    # served from memory
    assert FakeOllama.calls["embed"] == 1
    OllamaEmbedder(fake_ollama, "bge-m3", cache_path=cache).embed(["branch hours"])   # served from the file on disk
    assert FakeOllama.calls["embed"] == 1


def test_chat_request_format_and_result(fake_ollama):
    r = OllamaChat(fake_ollama, "qwen2.5:3b", num_ctx=2048).chat([{"role": "user", "content": "Hi"}], max_new_tokens=5)
    body = FakeOllama.calls["last_chat"]
    assert body["model"] == "qwen2.5:3b" and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 2048, "num_predict": 5, "seed": 0}
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ("Hello", 11, 7) and r.seconds >= 0


def test_local_calls_ignore_proxy_settings(fake_ollama, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")      # a dead proxy: would break any call that used it
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    assert OllamaChat(fake_ollama, "m").chat([{"role": "user", "content": "Hi"}]).text == "Hello"


def test_full_assistant_over_http(fake_ollama, tmp_path):
    bot = build_assistant(Settings(ollama_url=fake_ollama, use_embeddings=True, use_judge=True), cache_path=tmp_path / "emb.json")
    a = bot.answer("What is the late payment fee on a credit card?")
    assert (a.status, a.faq_ids, a.lang) == ("answered", [23], "en") and "500" in a.text
    # The fake echoes the English FAQ text, which is the wrong language for a Banglish question:
    # the language check must reject it twice and fall back to the original FAQ text.
    b = bot.answer("Personal loan er interest rate koto?")
    assert b.faq_ids == [19] and b.fallback == "verbatim_faq" and "12%" in b.text and b.retries == 1
    assert FakeOllama.calls["judge"] == 2 and FakeOllama.calls["judge_examples"] == 0   # examples are off by default


def test_run_eval_and_report_scripts_end_to_end(fake_ollama, tmp_path):
    # retrieval_min is pinned so this test does not depend on config/calibrated.json (written by calibrate.py)
    env = {**os.environ, "NIRBHOR_OLLAMA_URL": fake_ollama, "NIRBHOR_RETRIEVAL_MIN": "0.0", "NIRBHOR_USE_JUDGE": "true"}
    run = lambda *args: subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,  # noqa: E731
                                       env=env, cwd=ROOT, timeout=180)
    r = run(ROOT / "scripts" / "run_eval.py", "--gen-model", "fake", "--results-dir", tmp_path, "--tag", "e2e",
            "--hardware", "unit test", "--judge-examples")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    with open(tmp_path / "e2e" / "answers.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40 and {"manual_grade", "latency_ms", "correct_auto", "faq_used"} <= set(rows[0])
    metrics = json.loads((tmp_path / "e2e" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["total"] == 40 and metrics["meta"]["judge_examples"] is True
    # --judge-examples must reach the judge prompt on EVERY judge call of the run
    assert FakeOllama.calls["judge"] == 40 and FakeOllama.calls["judge_examples"] == 40   # one judge call per question

    # pretend a person graded two rows, then build the report and the failure analysis from the files on disk
    path = tmp_path / "e2e" / "answers.csv"
    for r_ in rows[:2]:
        r_["manual_grade"] = "1"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    assert run(ROOT / "scripts" / "make_report.py", "--results-dir", tmp_path).returncode == 0
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "e2e" in report and "2/2" in report                      # manual score picked up
    assert run(ROOT / "scripts" / "analyze_failures.py", tmp_path / "e2e").returncode == 0
    assert "Failure analysis" in (tmp_path / "e2e" / "failures.md").read_text(encoding="utf-8")
