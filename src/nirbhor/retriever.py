"""Finding the FAQ entry that matches a question.

Two signals are mixed:
  * dense   - embedding similarity (bge-m3 through Ollama, multilingual). Optional.
  * lexical - TF-IDF over English words. Bangla/Banglish questions are first expanded with English terms
              from glossary.py, so the keyword signal also works for them.
With no embedder the retriever runs on the lexical signal alone, so tests and a first demo work without any model.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .faq import FaqEntry
from .textutil import english_tokens


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedder:
    """Embeddings from a local Ollama server, cached on disk so repeated eval runs are fast."""

    def __init__(self, base_url: str, model: str, cache_path: Path | None = None, timeout: int = 120):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        self.cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        from .llm import local_session  # imported here so the keyword-only retriever needs no network library

        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        if missing:
            r = local_session().post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": missing, "keep_alive": "30m"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            for t, vec in zip(missing, r.json()["embeddings"]):
                self._cache[self._key(t)] = vec
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        mat = np.array([self._cache[self._key(t)] for t in texts], dtype=np.float32)
        return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)


@dataclass
class Hit:
    faq_id: int
    score: float
    dense: float
    lexical: float


class _TfIdf:
    def __init__(self, docs: list[list[str]]):
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        v = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, tokens: list[str]) -> list[float]:
        q = self._vec(tokens)
        return [sum(w * d.get(t, 0.0) for t, w in q.items()) for d in self.vecs]


class FaqRetriever:
    def __init__(self, faq: list[FaqEntry], embedder: Embedder | None = None, w_dense: float = 0.6):
        self.faq, self.embedder, self.w_dense = faq, embedder, w_dense
        self._tfidf = _TfIdf([english_tokens(f"{e.topic} {e.question} {e.answer}") for e in faq])
        self._dense_docs = None
        if embedder is not None:
            self._dense_docs = embedder.embed([f"{e.topic}. {e.question}" for e in faq])

    def search(self, query: str, expansion: str = "", extra_queries: list[str] | None = None, k: int = 3) -> list[Hit]:
        """Top-k FAQ entries. `expansion` = English glossary terms; `extra_queries` = e.g. an LLM translation."""
        lex_tokens = english_tokens(f"{query} {expansion} {' '.join(extra_queries or [])}")
        lexical = np.array(self._tfidf.scores(lex_tokens)) if lex_tokens else np.zeros(len(self.faq))

        if self.embedder is None:
            dense = np.zeros(len(self.faq))
            combined = lexical
        else:
            variants = [query] + ([f"{query} {expansion}"] if expansion else []) + list(extra_queries or [])
            q_vecs = self.embedder.embed(variants)
            dense = (q_vecs @ self._dense_docs.T).max(axis=0)
            combined = self.w_dense * dense + (1 - self.w_dense) * lexical

        order = np.argsort(-combined)[:k]
        return [Hit(self.faq[i].id, float(combined[i]), float(dense[i]), float(lexical[i])) for i in order]
