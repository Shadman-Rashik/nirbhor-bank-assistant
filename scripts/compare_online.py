"""Run the same 40 questions on an online model and put the result next to the local model.

Three steps
  1. API route (any OpenAI-compatible endpoint):
       set ONLINE_BASE_URL, ONLINE_API_KEY, ONLINE_MODEL, then:  python scripts/compare_online.py run
  2. No API key? Chat route:
       python scripts/compare_online.py export      -> paste results/online/prompt_for_chat.md into any online chat,
       save its answers into results/online/answers.csv (columns: id,answer,seconds), then:
       python scripts/compare_online.py grade
  3. python scripts/compare_online.py table --local results/test_qwen2.5_3b
       -> results/comparison.md (local vs online, side by side)

The online model gets the WHOLE FAQ in its prompt (no retrieval step) and the same rules, so the comparison is fair.
In the chat route you cannot measure speed precisely; type your stopwatch time into `seconds` or leave it blank.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from nirbhor.config import DATA_DIR, RESULTS_DIR
from nirbhor.faq import load_cases, load_faq
from nirbhor.lang import detect_language
from nirbhor.llm import OpenAICompatChat
from nirbhor.metrics import Row, grade, is_idk_text, summarize, summary_markdown
from nirbhor.prompts import online_system_prompt

ONLINE_DIR = RESULTS_DIR / "online"


def cmd_export() -> None:
    faq, cases = load_faq(DATA_DIR / "bank-faq.csv"), load_cases(DATA_DIR / "test-questions.csv", DATA_DIR / "gold.csv")
    ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    qs = "\n".join(f"{c.id}. {c.question}" for c in cases)
    (ONLINE_DIR / "prompt_for_chat.md").write_text(
        online_system_prompt(faq)
        + "\n\n---\nAnswer each customer question below as the assistant would. "
          "Output one line per question in the form `id | reply`, nothing else.\n\n" + qs + "\n",
        encoding="utf-8",
    )
    with open(ONLINE_DIR / "answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["id", "answer", "seconds"]] + [[c.id, "", ""] for c in cases])
    print(f"wrote {ONLINE_DIR / 'prompt_for_chat.md'} and a blank answers.csv")


def cmd_run() -> None:
    base, key, model = os.environ.get("ONLINE_BASE_URL"), os.environ.get("ONLINE_API_KEY"), os.environ.get("ONLINE_MODEL")
    if not (base and key and model):
        sys.exit("Set ONLINE_BASE_URL, ONLINE_API_KEY and ONLINE_MODEL first (see the top of this file).")
    faq, cases = load_faq(DATA_DIR / "bank-faq.csv"), load_cases(DATA_DIR / "test-questions.csv", DATA_DIR / "gold.csv")
    client, system = OpenAICompatChat(base, key, model), online_system_prompt(faq)
    ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ONLINE_DIR / "answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "answer", "seconds"])
        for c in cases:
            r = client.chat([{"role": "system", "content": system}, {"role": "user", "content": c.question}], max_new_tokens=400)
            w.writerow([c.id, r.text, round(r.seconds, 3)])
            print(f"id={c.id:<3} {r.seconds:5.2f}s", flush=True)
    (ONLINE_DIR / "model.txt").write_text(model, encoding="utf-8")
    cmd_grade()


def cmd_grade() -> dict:
    cases = {c.id: c for c in load_cases(DATA_DIR / "test-questions.csv", DATA_DIR / "gold.csv")}
    rows = []
    with open(ONLINE_DIR / "answers.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c, text = cases[int(r["id"])], (r["answer"] or "").strip()
            if not text:
                sys.exit(f"answers.csv has no answer for id {c.id}")
            status = "idk" if is_idk_text(text) else "answered"
            correct, facts, lm = grade(c, status, text, [], check_faq=False)
            sec = float(r["seconds"]) if (r.get("seconds") or "").strip() else 0.0
            rows.append(Row(c, detect_language(c.question), status, text, [], [], sec * 1000, correct, facts, lm))
    s = summarize(rows, detect_lang=False, with_retrieval=False)
    model = (ONLINE_DIR / "model.txt").read_text(encoding="utf-8").strip() if (ONLINE_DIR / "model.txt").exists() else "online model"
    md = summary_markdown(f"{model} (online) on the 40 given test questions", s)
    (ONLINE_DIR / "summary.md").write_text(md, encoding="utf-8")
    (ONLINE_DIR / "metrics.json").write_text(json.dumps({"meta": {"label": model}, "metrics": s}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    return s


def cmd_table(local_dir: Path) -> None:
    loc = json.loads((local_dir / "metrics.json").read_text(encoding="utf-8"))
    onl = json.loads((ONLINE_DIR / "metrics.json").read_text(encoding="utf-8"))
    L, O = loc["metrics"], onl["metrics"]

    def f(s, k1, k2):
        return f"{s[k1]}/{s[k2]}"

    rows = [
        ("Overall correct (of 40)", f(L, "correct", "total"), f(O, "correct", "total")),
        ("In-scope answered correctly", f(L, "in_scope_correct", "in_scope"), f(O, "in_scope_correct", "in_scope")),
        ("Correctly said \"I don't know\"", f(L, "idk_correct", "out_of_scope"), f(O, "idk_correct", "out_of_scope")),
        ("Median time per question (ms)", L["latency_ms"]["median"], O["latency_ms"]["median"] or "n/a"),
        ("95th percentile time (ms)", L["latency_ms"]["p95"], O["latency_ms"]["p95"] or "n/a"),
        ("Customer data leaves the bank's computer", "No", "Yes"),
    ]
    out = [f"| | Local: {loc['meta']['label']} | Online: {onl['meta']['label']} |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    (RESULTS_DIR / "comparison.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "run", "grade", "table"])
    ap.add_argument("--local", type=Path, help="results folder of the local run, for `table`")
    a = ap.parse_args()
    {"export": cmd_export, "run": cmd_run, "grade": cmd_grade}.get(a.command, lambda: cmd_table(a.local))()
