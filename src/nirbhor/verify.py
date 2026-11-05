"""Checks done in plain code on the model's reply: numbers must come from the FAQ, language must match."""
from __future__ import annotations

from dataclasses import dataclass

from .lang import banglish_score, script_ratio
from .textutil import digit_groups


@dataclass
class NumberCheck:
    ok: bool
    invented: list[str]


def check_numbers(reply: str, source_texts: list[str]) -> NumberCheck:
    """Every digit group in the reply (Latin or Bangla digits) must also appear in the FAQ text it was built from."""
    allowed: set[str] = set()
    for s in source_texts:
        allowed |= digit_groups(s)
    invented = sorted(digit_groups(reply) - allowed)
    return NumberCheck(not invented, invented)


def language_ok(reply: str, target: str) -> bool:
    """Is the reply written in the requested language/script?"""
    ratio = script_ratio(reply)
    if target == "bn":
        return ratio >= 0.5
    if target == "banglish":
        return ratio < 0.05 and banglish_score(reply) >= 1
    return ratio < 0.05 and banglish_score(reply) < 1.5
