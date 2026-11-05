"""Loading the FAQ and the question sets (UTF-8 with BOM safe)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaqEntry:
    id: int
    topic: str
    question: str
    answer: str


@dataclass(frozen=True)
class Case:
    """One evaluation question with its gold label."""
    id: int
    question: str
    lang: str                      # gold language: en | bn | banglish
    expected_faq_id: int | None    # None = the FAQ does not cover it, correct reply is "I don't know"
    key_facts: tuple[tuple[str, ...], ...]  # every group must match; any alternative in a group matches
    notes: str = ""


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_faq(path: str | Path) -> list[FaqEntry]:
    return [
        FaqEntry(int(r["id"]), r["topic"].strip(), r["question"].strip(), r["answer"].strip())
        for r in _read_rows(Path(path))
    ]


def parse_key_facts(raw: str) -> tuple[tuple[str, ...], ...]:
    """'12;a|b' -> (('12',), ('a','b')). Groups split on ';', alternatives on '|'."""
    raw = (raw or "").strip()
    if not raw:
        return ()
    return tuple(tuple(alt.strip() for alt in grp.split("|") if alt.strip()) for grp in raw.split(";"))


def load_cases(questions_path: str | Path, gold_path: str | Path | None = None) -> list[Case]:
    """Questions + gold labels. Gold can live in the same file (held-out set) or in a separate one."""
    rows = _read_rows(Path(questions_path))
    gold = {int(r["id"]): r for r in _read_rows(Path(gold_path))} if gold_path else {}
    cases = []
    for r in rows:
        g = gold.get(int(r["id"]), r)
        raw_faq = (g.get("expected_faq_id") or "").strip().upper()
        cases.append(
            Case(
                id=int(r["id"]),
                question=r["question"].strip(),
                lang=(g.get("lang") or "").strip(),
                expected_faq_id=None if raw_faq in ("", "NONE") else int(raw_faq),
                key_facts=parse_key_facts(g.get("key_facts", "")),
                notes=(g.get("notes") or "").strip(),
            )
        )
    return cases
