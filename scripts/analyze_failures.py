"""Sort every wrong answer of a run into a cause, so the note can say WHY it failed (not just how many).

  python scripts/analyze_failures.py results/test_qwen2.5_3b            # one run
  python scripts/analyze_failures.py                                    # every run under results/

Writes failures.md next to answers.csv. Uses the automatic grade; if you filled the manual_grade column
(1 = right, 0 = wrong), rows you graded 0 are analysed and rows you graded 1 are skipped, so your judgement wins.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from nirbhor.config import RESULTS_DIR

REFUSAL_REASON = {
    "retrieval_below_threshold": "Refused by the retrieval gate (best match score too low)",
    "judge_rejected": "Refused by the strict check (judge said NO)",
    "llm_abstain": "Model replied NO_ANSWER",
}


def is_wrong(r: dict) -> bool:
    manual = (r.get("manual_grade") or "").strip()
    if manual in ("0", "1"):
        return manual == "0"
    return r["correct_auto"] != "True"


def category(r: dict) -> str:
    exp, status = r["expected_faq"], r["status"]
    if r["reason"] == "model_error":
        return "Server/model error (Ollama not reachable, crashed or timed out): not a quality failure, re-run"
    if exp == "NONE":
        return "Answered a question the FAQ does not cover"
    if status == "idk":
        return REFUSAL_REASON.get(r["reason"], "Refused (other reason)")
    used = [x for x in r["faq_used"].split(",") if x]
    if exp not in used:
        in_top3 = exp in r["top3"].split(",")
        return ("Wrong FAQ entry used, but the right one was in the top 3 (ordering or judge)" if in_top3
                else "Wrong FAQ entry used: retrieval did not find the right one")
    if r["lang_ok"] == "False":
        return ("Reply fell back to the English FAQ text (translation failed the number/language check twice)"
                if r["fallback"] else "Reply is not in the customer's language")
    if r["facts_ok"] == "False":
        return "Reply is missing a key fact (or has it wrong)"
    return "Other: automatic check says right, manual grade says wrong (read the answer)"


def analyse(run_dir: Path) -> Path:
    with open(run_dir / "answers.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    wrong = [r for r in rows if is_wrong(r)]
    counts = Counter(category(r) for r in wrong)
    lines = [f"# Failure analysis: {run_dir.name}", "",
             f"{len(wrong)} wrong out of {len(rows)} questions.", "",
             "| Cause | Count |", "|---|---|"]
    lines += [f"| {c} | {n} |" for c, n in counts.most_common()] or ["| (none) | 0 |"]
    lines += ["", "## Each wrong answer", ""]
    for r in wrong:
        ans = " ".join(r["answer"].split())
        lines += [
            f"### Question {r['id']} ({r['lang_gold']}): {r['question']}",
            f"- Cause: {category(r)}",
            f"- Expected FAQ entry: {r['expected_faq']} | used: {r['faq_used'] or '-'} | top 3 by search: {r['top3']}",
            f"- Status: {r['status']} ({r['reason']})",
            f"- Reply: {ans[:300]}{'...' if len(ans) > 300 else ''}",
            "",
        ]
    out = run_dir / "failures.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{run_dir.name}: {len(wrong)} wrong")
    for c, n in counts.most_common():
        print(f"   {n:>2}  {c}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    a = ap.parse_args()
    dirs = a.run_dirs or sorted(p.parent for p in RESULTS_DIR.glob("*/answers.csv"))
    if not dirs:
        sys.exit("No results yet. Run scripts/run_eval.py first.")
    for d in dirs:
        analyse(d)


if __name__ == "__main__":
    main()
