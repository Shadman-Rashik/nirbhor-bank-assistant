"""Mistake handling: the assistant must not crash, and a crash must never be scored as a correct answer."""
import os
import subprocess
import sys
from pathlib import Path

from conftest import ScriptedLLM
from test_ollama_http import fake_ollama  # noqa: F401  (pytest fixture)

from nirbhor.config import Settings
from nirbhor.pipeline import Assistant, MAX_QUESTION_CHARS
from nirbhor.retriever import FaqRetriever
from nirbhor.templates import IDK

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class BrokenLLM:
    model = "broken"

    def chat(self, messages, max_new_tokens=200, temperature=0.0):
        raise ConnectionError("Ollama is down")


def test_model_failure_becomes_a_handover_not_a_crash(make_assistant):
    a = make_assistant(BrokenLLM()).answer("What is the late payment fee on a credit card?")
    assert (a.status, a.reason, a.text) == ("idk", "model_error", IDK["en"])
    assert "ConnectionError" in a.error and "Ollama is down" in a.error


def test_embedding_failure_is_also_handled(faq):
    class Boom:
        def search(self, *args, **kwargs):
            raise TimeoutError("embedding server timed out")

    a = Assistant(faq, Boom(), ScriptedLLM(), Settings(use_embeddings=False)).answer("Is there a fee to receive remittance?")
    assert a.reason == "model_error" and "timed out" in a.error and a.text == IDK["en"]


def test_empty_and_huge_input(make_assistant):
    bot = make_assistant(ScriptedLLM())
    assert bot.answer("   ").reason == "empty_input" and bot.answer(None).reason == "empty_input"
    a = bot.answer("loan interest " * 2000)                       # must not blow up
    assert len(a.question) <= MAX_QUESTION_CHARS


def test_eval_scores_a_crash_as_wrong_even_for_dont_know_questions(faq, test_cases):
    import run_eval

    bot = Assistant(faq, FaqRetriever(faq, None), BrokenLLM(), Settings(use_embeddings=False, retrieval_min=0.0))
    rows = run_eval.evaluate(bot, test_cases)
    assert not any(r.correct for r in rows)                        # incl. the 10 out-of-scope ones that look like "I don't know"
    assert all(r.reason == "model_error" for r in rows)


def _run(script, *args, env=None):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT, env={**os.environ, **(env or {})}, timeout=300)


def test_run_eval_explains_a_missing_server_without_a_stack_trace():
    r = _run("run_eval.py", "--gen-model", "x", env={"NIRBHOR_OLLAMA_URL": "http://127.0.0.1:9"})
    assert r.returncode != 0 and "Cannot reach the model server" in r.stderr and "ollama pull" in r.stderr
    assert "Traceback" not in r.stderr


def test_run_all_end_to_end_and_missing_model(fake_ollama, tmp_path):  # noqa: F811
    env = {"NIRBHOR_OLLAMA_URL": fake_ollama, "NIRBHOR_RETRIEVAL_MIN": "0.0"}
    r = _run("run_all.py", "--models", "fake", "--results-dir", tmp_path, "--no-calibrate", "--hardware", "unit test", env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert {"test_fake", "heldout_fake", "ablation_with_judge", "ablation_keywords_only", "ablation_judge_examples"} <= runs
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "test_fake" / "failures.md").exists()

    r = _run("run_all.py", "--models", "not-installed", "--results-dir", tmp_path / "x", "--no-calibrate", env=env)
    assert r.returncode == 2 and "ollama pull not-installed" in r.stdout
