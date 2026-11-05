"""Scoring. One definition of "correct" shared by the local assistant and the online comparison.

Out of scope (gold = NONE): correct only if the reply is an "I don't know / connect to a person" reply.
In scope: correct only if it is not a refusal, contains the key facts, and is in the customer's language.
(For the local assistant it must also be built from the right FAQ entry.)
This is an automatic, lenient check. The CSV written by run_eval.py has a manual_grade column for a human pass.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

from .faq import Case
from .textutil import normalize_text
from .verify import language_ok

_IDK_PATTERNS = [
    r"don'?t know", r"do not know", r"not (?:covered|mentioned|available|in (?:our|the) faq)",
    r"(?:cannot|can'?t|unable to) (?:find|answer|help)", r"no information", r"connect you to",
    r"\b(?:human|person|representative|agent|team member)\b",
    r"জানি না", r"তথ্য (?:নেই|নাই)", r"যুক্ত করছি", r"সংযুক্ত", r"প্রতিনিধি", r"উত্তর দিতে পারছি না",
    r"jani na", r"tottho nei", r"connect kore", r"prothinidhi", r"manusher",
]
_IDK_RE = re.compile("|".join(_IDK_PATTERNS), re.I)


def is_idk_text(text: str) -> bool:
    return bool(_IDK_RE.search(text))


def facts_ok(text: str, key_facts: tuple[tuple[str, ...], ...]) -> bool:
    """Every group must have at least one alternative present (case-insensitive, digits/commas normalised)."""
    hay = normalize_text(text).lower()
    return all(any(normalize_text(alt).lower() in hay for alt in group) for group in key_facts)


@dataclass
class Row:
    case: Case
    lang_pred: str
    status: str                 # answered | idk
    text: str
    faq_ids: list[int]
    top3_ids: list[int]
    latency_ms: float
    correct: bool = False
    facts: bool | None = None
    lang_match: bool | None = None
    reason: str = ""


def grade(case: Case, status: str, text: str, faq_ids: list[int], check_faq: bool = True) -> tuple[bool, bool | None, bool | None]:
    """-> (correct, facts_ok, language_ok)"""
    if case.expected_faq_id is None:
        return status == "idk", None, None
    if status != "answered":
        return False, False, None
    f = facts_ok(text, case.key_facts)
    l = language_ok(text, case.lang)
    faq_good = (case.expected_faq_id in faq_ids) if check_faq else True
    return (f and l and faq_good), f, l


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(rows: list[Row], detect_lang: bool = True, with_retrieval: bool = True) -> dict:
    ins = [r for r in rows if r.case.expected_faq_id is not None]
    out = [r for r in rows if r.case.expected_faq_id is None]
    lat = [r.latency_ms for r in rows if r.latency_ms > 0]
    by_lang: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_lang[r.case.lang].append(r)
    s = {
        "total": len(rows),
        "correct": sum(r.correct for r in rows),
        "in_scope": len(ins),
        "in_scope_correct": sum(r.correct for r in ins),
        "out_of_scope": len(out),
        "idk_correct": sum(r.correct for r in out),            # said "I don't know" when it should
        "false_idk": sum(r.status == "idk" for r in ins),      # refused a question the FAQ covers
        "false_answer": sum(r.status == "answered" for r in out),  # answered something the FAQ doesn't cover
        "by_language": {k: {"correct": sum(r.correct for r in v), "total": len(v)} for k, v in sorted(by_lang.items())},
        "latency_ms": {
            "mean": round(statistics.fmean(lat), 1) if lat else 0,
            "median": round(statistics.median(lat), 1) if lat else 0,
            "p95": round(percentile(lat, 95), 1),
            "max": round(max(lat), 1) if lat else 0,
        },
    }
    if with_retrieval:
        s["retrieval_top1"] = sum(r.case.expected_faq_id in r.top3_ids[:1] for r in ins)
        s["retrieval_top3"] = sum(r.case.expected_faq_id in r.top3_ids for r in ins)
    if detect_lang:
        s["language_detected_correctly"] = sum(r.lang_pred == r.case.lang for r in rows)
    return s


def summary_markdown(title: str, s: dict, extra: dict | None = None) -> str:
    lines = [f"### {title}", "", "| Metric | Result |", "|---|---|"]
    lines.append(f"| Overall correct | {_pct(s['correct'], s['total'])} |")
    lines.append(f"| In-scope answered correctly | {_pct(s['in_scope_correct'], s['in_scope'])} |")
    lines.append(f"| Correctly said \"I don't know\" | {_pct(s['idk_correct'], s['out_of_scope'])} |")
    lines.append(f"| Refused a question the FAQ covers (false \"I don't know\") | {s['false_idk']} |")
    lines.append(f"| Answered a question the FAQ does not cover | {s['false_answer']} |")
    for lang, v in s["by_language"].items():
        lines.append(f"| Correct, {lang} questions | {_pct(v['correct'], v['total'])} |")
    if "language_detected_correctly" in s:
        lines.append(f"| Language detected correctly | {_pct(s['language_detected_correctly'], s['total'])} |")
    if "retrieval_top1" in s:
        lines.append(f"| Right FAQ entry ranked #1 / in top 3 | {_pct(s['retrieval_top1'], s['in_scope'])} / {_pct(s['retrieval_top3'], s['in_scope'])} |")
    lat = s["latency_ms"]
    lines.append(f"| Latency per question (ms): mean / median / p95 / max | {lat['mean']} / {lat['median']} / {lat['p95']} / {lat['max']} |")
    for k, v in (extra or {}).items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"
