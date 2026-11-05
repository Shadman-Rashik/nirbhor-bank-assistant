from pathlib import Path

import pytest

from nirbhor.config import DATA_DIR, Settings
from nirbhor.faq import load_cases, load_faq
from nirbhor.llm import LLMResult
from nirbhor.pipeline import Assistant
from nirbhor.retriever import FaqRetriever


class ScriptedLLM:
    """Stands in for the small model. `script` maps a marker found in the prompt to a canned reply."""

    model = "scripted"

    def __init__(self, judge="YES", replies=None):
        self.judge, self.replies, self.calls = judge, list(replies or []), []

    def chat(self, messages, max_new_tokens=200, temperature=0.0):
        system = messages[0]["content"]
        self.calls.append(system[:40])
        if "You check whether a bank FAQ entry" in system:
            return LLMResult(self.judge, 0.0)
        return LLMResult(self.replies.pop(0) if self.replies else "NO_ANSWER", 0.0)


@pytest.fixture(scope="session")
def faq():
    return load_faq(DATA_DIR / "bank-faq.csv")


@pytest.fixture(scope="session")
def test_cases():
    return load_cases(DATA_DIR / "test-questions.csv", DATA_DIR / "gold.csv")


@pytest.fixture(scope="session")
def heldout_cases():
    return load_cases(DATA_DIR / "heldout.csv")


@pytest.fixture()
def make_assistant(faq):
    def _make(llm, **settings):
        settings.setdefault("use_judge", True)   # the pipeline tests exercise the judge step; the product default is off
        s = Settings(use_embeddings=False, **settings)
        return Assistant(faq, FaqRetriever(faq, None), llm, s)

    return _make
