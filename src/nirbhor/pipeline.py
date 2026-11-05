"""The assistant: a fixed sequence of steps. Every step that can say "I don't know" does so with a fixed reply.

  1. detect language           (code)
  2. retrieve FAQ candidates   (embeddings + keywords, glossary for Bangla/Banglish)
  3. gate: score too low       -> "I don't know"   (no model call)
  4. judge: does the entry answer this question?   NO -> "I don't know"   (small model, one word)
  5. generate the reply in the customer's language from that one FAQ entry   (NO_ANSWER -> "I don't know")
  6. verify in code: every number is in the FAQ entry, language is right; retry once; else show the FAQ text
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from . import glossary
from .config import Settings
from .faq import FaqEntry
from .lang import detect_language
from .llm import ChatClient
from .prompts import NO_ANSWER, build_judge_messages, system_prompt, translate_messages, user_prompt
from .retriever import FaqRetriever, Hit
from .templates import FALLBACK_PREFIX, IDK, LANG_NAME
from .verify import check_numbers, language_ok

_THINK = re.compile(r"<think>.*?</think>", re.S)
_LABEL = re.compile(r"^(reply|answer|response)\s*:\s*", re.I)
log = logging.getLogger(__name__)
MAX_QUESTION_CHARS = 500   # longer input is cut: a bank FAQ question is never this long, and it protects the model's context


@dataclass
class Answer:
    question: str
    lang: str
    status: str                      # "answered" | "idk"
    text: str
    reason: str                      # why: answered / retrieval_below_threshold / judge_rejected / llm_abstain ...
    faq_ids: list[int] = field(default_factory=list)   # FAQ entries the reply is built from ([] for idk)
    hits: list[dict] = field(default_factory=list)     # top retrieved candidates with scores (diagnostics)
    judge: list[dict] = field(default_factory=list)
    retries: int = 0
    error: str | None = None         # filled when reason == "model_error" (server down, timeout, ...)
    fallback: str | None = None      # "verbatim_faq" when the translated reply failed verification twice
    timings_ms: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


def _clean(text: str) -> str:
    text = _THINK.sub("", text).strip()
    text = _LABEL.sub("", text).strip()
    return text.strip().strip('"“”').strip()


class Assistant:
    def __init__(self, faq: list[FaqEntry], retriever: FaqRetriever, llm: ChatClient | None, settings: Settings):
        self.faq, self.retriever, self.llm, self.s = faq, retriever, llm, settings
        self.by_id = {e.id: e for e in faq}

    # ------------------------------------------------------------------ public
    def warmup(self) -> float:
        """Load the model into memory so the first real question isn't charged for loading."""
        if self.llm is None:
            return 0.0
        t0 = time.perf_counter()
        self.llm.chat([{"role": "user", "content": "Hi"}], max_new_tokens=2)
        return time.perf_counter() - t0

    def answer(self, question: str) -> Answer:
        """Never raises: any failure (model down, timeout, bad input) becomes a safe hand-over to a person."""
        t0 = time.perf_counter()
        q = " ".join(str(question or "").split())[:MAX_QUESTION_CHARS]
        if not q:
            return Answer("", "en", "idk", IDK["en"], "empty_input", timings_ms={"total": 0.0})
        try:
            return self._answer(q)
        except Exception as e:  # noqa: BLE001 - deliberate: a customer must never see a stack trace
            log.warning("answer failed: %s: %s", type(e).__name__, e)
            lang = detect_language(q)
            return Answer(q, lang, "idk", IDK[lang], "model_error", error=f"{type(e).__name__}: {e}",
                          timings_ms={"total": (time.perf_counter() - t0) * 1000})

    def _answer(self, q: str) -> Answer:
        t0 = time.perf_counter()
        lang = detect_language(q)
        ms: dict = {}

        # 2. retrieve -------------------------------------------------------
        t = time.perf_counter()
        extra: list[str] = []
        if self.llm is not None and self.s.use_query_translation and lang != "en":
            extra.append(_clean(self.llm.chat(translate_messages(q), max_new_tokens=60).text))
        expansion = glossary.expand(q) if lang != "en" else ""
        hits = self.retriever.search(q, expansion, extra, k=self.s.top_k)
        ms["retrieval"] = (time.perf_counter() - t) * 1000
        hit_log = [{"faq_id": h.faq_id, "score": round(h.score, 3), "dense": round(h.dense, 3),
                    "lexical": round(h.lexical, 3)} for h in hits]

        def done(status, text, reason, **kw) -> Answer:
            ms["total"] = (time.perf_counter() - t0) * 1000
            return Answer(q, lang, status, text, reason, hits=hit_log, timings_ms=ms, **kw)

        def idk(reason, **kw) -> Answer:
            return done("idk", IDK[lang], reason, **kw)

        # 3. gate -----------------------------------------------------------
        if not hits or hits[0].score < self.s.retrieval_min:
            return idk("retrieval_below_threshold")
        if self.llm is None:  # retrieval-only mode (no model installed): hand back the FAQ text as it is
            return done("answered", self.by_id[hits[0].faq_id].answer, "retrieval_only", faq_ids=[hits[0].faq_id])

        # 4. judge ----------------------------------------------------------
        chosen: Hit | None = hits[0]
        judge_log: list[dict] = []
        if self.s.use_judge:
            t = time.perf_counter()
            chosen = None
            for h in hits[:2]:
                if h.score < self.s.retrieval_min or (h is not hits[0] and hits[0].score - h.score > 0.10):
                    break
                verdict = self._judge(self.by_id[h.faq_id], q)
                judge_log.append({"faq_id": h.faq_id, "yes": verdict})
                if verdict:
                    chosen = h
                    break
            ms["judge"] = (time.perf_counter() - t) * 1000
            if chosen is None:
                return idk("judge_rejected", judge=judge_log)

        # 5-6. generate + verify -------------------------------------------
        t = time.perf_counter()
        entry = self.by_id[chosen.faq_id]
        text, status, retries, fallback = self._generate(lang, entry, q)
        ms["generate"] = (time.perf_counter() - t) * 1000
        if status == "abstain":
            return idk("llm_abstain", judge=judge_log, retries=retries)
        return done("answered", text, "answered", faq_ids=[entry.id], judge=judge_log, retries=retries, fallback=fallback)

    # ----------------------------------------------------------------- helpers
    def _judge(self, entry: FaqEntry, question: str) -> bool:
        out = self.llm.chat(build_judge_messages(entry, question, self.s.judge_examples), max_new_tokens=4, temperature=0.0).text
        return _clean(out).upper().startswith("YES")

    def _max_tokens(self, lang: str) -> int:
        return {"en": self.s.max_new_tokens_en, "banglish": self.s.max_new_tokens_banglish,
                "bn": self.s.max_new_tokens_bn}[lang]

    def _generate(self, lang: str, entry: FaqEntry, question: str):
        """Returns (text, status, retries, fallback). status is 'ok' or 'abstain'."""
        problem = ""
        for attempt in range(2):
            messages = [
                {"role": "system", "content": system_prompt(lang, strict=attempt > 0, problem=problem)},
                {"role": "user", "content": user_prompt(entry, question)},
            ]
            raw = self.llm.chat(messages, max_new_tokens=self._max_tokens(lang), temperature=self.s.temperature).text
            text = _clean(raw)
            if NO_ANSWER in text.upper().replace(" ", "_"):
                return "", "abstain", attempt, None
            nums = check_numbers(text, [entry.answer, entry.question, question])
            if text and nums.ok and language_ok(text, lang):
                return text, "ok", attempt, None
            problem = (
                f"it contained numbers that are not in the FAQ entry ({', '.join(nums.invented)})"
                if not nums.ok
                else f"it was not written in {LANG_NAME[lang]} as required"
            )
        return FALLBACK_PREFIX[lang] + entry.answer, "ok", 1, "verbatim_faq"
