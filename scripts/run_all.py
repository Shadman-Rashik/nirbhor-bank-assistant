"""Run the whole evaluation with ONE command, then build the failure analysis and the results table.

  python scripts/run_all.py --models qwen2.5:3b gemma3:4b --hardware "GTX 1650 4GB, Windows"

What it does, in order (each step is a normal script you can also run alone):
  1. calibrate     picks the retrieval threshold on the held-out questions and saves config/calibrated.json
  2. models        the 40 given questions on every model you list (model comparison)
  3. held-out      my 40 extra questions on the first model (checks for over-fitting)
  4. ablations     first model with: no strict check / keyword search only / worked-example judge prompt
  5. analysis      failures.md for every run + results/REPORT.md (one table)
If a step fails it is reported at the end and the remaining steps still run. Re-running overwrites the same folders.
Useful: --dry-run (print the plan only), --no-ablations, --no-calibrate, --results-dir PATH.
"""
import _bootstrap  # noqa: F401

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from nirbhor.config import RESULTS_DIR, load_settings
from nirbhor.llm import local_session

SCRIPTS = Path(__file__).resolve().parent


def installed_models(base_url: str) -> set[str]:
    r = local_session().get(f"{base_url.rstrip('/')}/api/tags", timeout=15)
    r.raise_for_status()
    names = {m["name"] for m in r.json().get("models", [])}
    return names | {n[: -len(":latest")] for n in names if n.endswith(":latest")}


def run_step(label: str, script: str, *args) -> bool:
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print(f"\n{'=' * 78}\n>>> {label}\n    {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    ok = subprocess.run(cmd).returncode == 0
    print(f"<<< {label}: {'done' if ok else 'FAILED'} in {time.perf_counter() - t0:.0f}s", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen2.5:3b"], help="Ollama tags; the first one is used for held-out and ablations")
    ap.add_argument("--hardware", default="unspecified", help='written into every result, e.g. "Colab T4 GPU"')
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    s = load_settings()
    base = ["--hardware", a.hardware, "--results-dir", a.results_dir]
    main_model = a.models[0]
    plan: list[tuple[str, str, list]] = []
    if not a.no_calibrate:
        plan.append(("Calibrate the retrieval threshold (held-out questions)", "calibrate.py", ["--write"]))
    for m in a.models:
        plan.append((f"40 given questions: {m}", "run_eval.py", ["--gen-model", m, *base]))
    plan.append((f"Held-out questions: {main_model}", "run_eval.py", ["--set", "heldout", "--gen-model", main_model, *base]))
    if not a.no_ablations:
        plan += [
            # the default (final) configuration has the strict check OFF; the ablations switch things around it
            (f"Ablation, strict check ON: {main_model}", "run_eval.py", ["--gen-model", main_model, "--judge", "--tag", "ablation_with_judge", *base]),
            (f"Ablation, keyword search only: {main_model}", "run_eval.py", ["--gen-model", main_model, "--no-embeddings", "--tag", "ablation_keywords_only", *base]),
            (f"Ablation, strict check ON with worked examples: {main_model}", "run_eval.py", ["--gen-model", main_model, "--judge", "--judge-examples", "--tag", "ablation_judge_examples", *base]),
        ]
    plan += [("Failure analysis (every run folder)", "analyze_failures.py", []),   # folders are listed at run time
             ("Results table", "make_report.py", ["--results-dir", a.results_dir])]

    if a.dry_run:
        for i, (label, script, args) in enumerate(plan, 1):
            print(f"{i:>2}. {label}\n      python scripts/{script} {' '.join(map(str, args))}")
        return 0

    # Check the server and the models first, so a missing model fails in 1 second instead of after 10 minutes.
    try:
        have = installed_models(s.ollama_url)
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach Ollama at {s.ollama_url}: {type(e).__name__}: {e}\nStart Ollama first (on Windows: open the Ollama app).")
        return 2
    needed = [*a.models, s.embed_model]
    missing = [m for m in needed if m not in have]
    if missing:
        print("These models are not installed yet: " + ", ".join(missing))
        for m in missing:
            print(f"    ollama pull {m}")
        print(f"(installed now: {', '.join(sorted(have)) or 'none'})")
        return 2

    a.results_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    t0 = time.perf_counter()
    for label, script, args in plan:
        if script == "analyze_failures.py":   # analyse every run folder that has answers
            dirs = sorted(p.parent for p in a.results_dir.glob("*/answers.csv"))
            args = dirs
        if not run_step(label, script, *args):
            failed.append(label)
    print(f"\nAll steps finished in {(time.perf_counter() - t0) / 60:.1f} min.")
    if failed:
        print("FAILED steps (fix and re-run):\n  - " + "\n  - ".join(failed))
    print(f"\nNext: open the answers.csv inside each run folder of {a.results_dir}, fill the manual_grade column (1 = right, 0 = wrong),\n"
          "then run:  python scripts/analyze_failures.py   and   python scripts/make_report.py\n"
          "and copy the numbers into README.md and docs/ONE_PAGE_NOTE.md.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
