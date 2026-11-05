"""Run the assistant on a question set and write answers + metrics.

Examples
  python scripts/run_eval.py --gen-model qwen2.5:3b --hardware "GTX 1650 4GB"
  python scripts/run_eval.py --gen-model qwen2.5:3b --gen-model gemma3:4b --gen-model llama3.2:3b
  python scripts/run_eval.py --set heldout --gen-model qwen2.5:3b          # your own 40 extra questions
  python scripts/run_eval.py --retrieval-only --no-embeddings               # no model installed: keyword retrieval only
  python scripts/run_eval.py --gen-model qwen2.5:3b --no-judge --tag ablation_no_judge
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

from nirbhor.config import DATA_DIR, RESULTS_DIR, load_settings
from nirbhor.factory import build_assistant
from nirbhor.faq import load_cases
from nirbhor.metrics import Row, grade, summarize, summary_markdown


def evaluate(assistant, cases, check_faq: bool = True) -> list[Row]:
    rows = []
    for i, c in enumerate(cases, 1):
        a = assistant.answer(c.question)
        correct, facts, lang_match = grade(c, a.status, a.text, a.faq_ids, check_faq)
        if a.reason == "model_error":   # a crash is never a correct "I don't know"
            correct, facts, lang_match = False, None, None
        rows.append(Row(c, a.lang, a.status, a.text, a.faq_ids, [h["faq_id"] for h in a.hits], a.total_ms,
                        correct, facts, lang_match, a.reason))
        rows[-1].raw = a  # keep the full Answer for the CSV
        mark = "ERR" if a.reason == "model_error" else ("ok " if correct else "BAD")
        print(f"[{i:>2}/{len(cases)}] {mark} id={c.id:<3} {a.lang:<8} {a.status:<8} {a.reason:<26} {a.total_ms:>7.0f} ms", flush=True)
    return rows


def write_outputs(out_dir: Path, rows: list[Row], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["id", "lang_gold", "lang_detected", "question", "expected_faq", "status", "reason", "faq_used",
              "top3", "answer", "facts_ok", "lang_ok", "correct_auto", "latency_ms", "retrieval_ms", "judge_ms",
              "generate_ms", "retries", "fallback", "error", "manual_grade", "manual_note"]
    with open(out_dir / "answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            a = r.raw
            w.writerow({
                "id": r.case.id, "lang_gold": r.case.lang, "lang_detected": r.lang_pred, "question": r.case.question,
                "expected_faq": r.case.expected_faq_id or "NONE", "status": r.status, "reason": r.reason,
                "faq_used": ",".join(map(str, r.faq_ids)), "top3": ",".join(map(str, r.top3_ids)), "answer": r.text,
                "facts_ok": "" if r.facts is None else r.facts, "lang_ok": "" if r.lang_match is None else r.lang_match,
                "correct_auto": r.correct, "latency_ms": round(r.latency_ms),
                "retrieval_ms": round(a.timings_ms.get("retrieval", 0)), "judge_ms": round(a.timings_ms.get("judge", 0)),
                "generate_ms": round(a.timings_ms.get("generate", 0)), "retries": a.retries,
                "fallback": a.fallback or "", "error": a.error or "", "manual_grade": "", "manual_note": "",
            })
    s = summarize(rows)
    fallbacks = sum(1 for r in rows if r.raw.fallback)
    s["fallback_to_faq_text"] = fallbacks
    errors = sum(1 for r in rows if r.reason == "model_error")
    s["model_errors"] = errors
    (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": s}, indent=2, ensure_ascii=False), encoding="utf-8")
    md = summary_markdown(f"{meta['label']} on {meta['question_set']}", s, {
        "Replies that fell back to the FAQ text (failed number/language check twice)": fallbacks,
        "Questions that hit a server/model error (scored wrong)": errors,
        "Hardware": meta["hardware"],
    })
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    return s


def connect(settings, retrieval_only: bool, results_root: Path):
    """Build the assistant and load the model; stop with a plain-language message (no stack trace) if the server is not ready."""
    try:
        assistant = build_assistant(settings, with_llm=not retrieval_only, cache_path=results_root / ".embed_cache.json")
        return assistant, assistant.warmup()
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"\nCannot reach the model server at {settings.ollama_url}: {type(e).__name__}: {e}\n"
            f"Is Ollama running? Did you run:  ollama pull {settings.gen_model}   and   ollama pull {settings.embed_model}  ?\n"
            "(To test without any model: add --retrieval-only --no-embeddings)"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["test", "heldout"], default="test", help="test = the 40 given questions")
    p.add_argument("--gen-model", action="append", help="Ollama model tag; repeat to compare several models")
    p.add_argument("--embed-model")
    p.add_argument("--no-embeddings", action="store_true", help="keyword-only retrieval")
    p.add_argument("--no-judge", action="store_true", help="strict check off (this is the default configuration)")
    p.add_argument("--judge", action="store_true", help="turn the strict yes/no check on (an ablation; off in the final configuration)")
    p.add_argument("--translate-query", action="store_true", help="let the model translate bn/banglish queries first")
    p.add_argument("--tau", type=float, help="retrieval threshold (default: config/calibrated.json or built-in)")
    p.add_argument("--retrieval-only", action="store_true", help="skip the generation model entirely")
    p.add_argument("--hardware", default="unspecified", help='e.g. "GTX 1650 4GB, Windows 11"')
    p.add_argument("--tag", help="output folder name under results/")
    p.add_argument("--judge-examples", action="store_true", help="add worked examples to the judge prompt (A/B test it)")
    p.add_argument("--results-dir", type=Path, help="where to write results (default: results/)")
    args = p.parse_args()
    results_root = args.results_dir or RESULTS_DIR

    if args.set == "test":
        cases, label_set = load_cases(DATA_DIR / "test-questions.csv", DATA_DIR / "gold.csv"), "the 40 given test questions"
    else:
        cases, label_set = load_cases(DATA_DIR / "heldout.csv"), "the self-written held-out questions"

    models = args.gen_model or [load_settings().gen_model]
    if args.retrieval_only:
        models = ["(no generation model)"]
    for model in models:
        s = load_settings(
            gen_model=None if args.retrieval_only else model, embed_model=args.embed_model, retrieval_min=args.tau,
            use_embeddings=False if args.no_embeddings else None, use_judge=False if args.no_judge else (True if args.judge else None),
            use_query_translation=True if args.translate_query else None,
            judge_examples=True if args.judge_examples else None,
        )
        assistant, warm = connect(s, args.retrieval_only, results_root)
        print(f"\n=== {model} | embeddings={'off' if not s.use_embeddings else s.embed_model} | judge={s.use_judge} | tau={s.retrieval_min} ===")
        tag = args.tag or re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.set}_{model}")
        meta = {
            "label": model, "question_set": label_set, "tag": tag, "date": datetime.now().isoformat(timespec="seconds"),
            "gen_model": None if args.retrieval_only else s.gen_model, "embed_model": s.embed_model if s.use_embeddings else None,
            "use_judge": s.use_judge, "judge_examples": s.judge_examples, "use_query_translation": s.use_query_translation, "retrieval_min": s.retrieval_min,
            "w_dense": s.w_dense, "hardware": args.hardware, "model_load_warmup_s": round(warm, 1),
            "python": platform.python_version(), "os": platform.platform(),
        }
        summary = write_outputs(results_root / tag, evaluate(assistant, cases, check_faq=True), meta)
        print(f"\n{summary['correct']}/{summary['total']} correct | in-scope {summary['in_scope_correct']}/{summary['in_scope']} | "
              f"'I don't know' {summary['idk_correct']}/{summary['out_of_scope']} | median {summary['latency_ms']['median']} ms")
        if summary.get("model_errors"):
            print(f"WARNING: {summary['model_errors']} question(s) hit a server/model error and were scored WRONG. "
                  "Fix the server (see the 'error' column in answers.csv) and re-run before trusting these numbers.")
        print(f"written to results/{tag}/  (open answers.csv and fill the manual_grade column)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
