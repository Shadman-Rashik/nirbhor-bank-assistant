"""Choose the retrieval threshold on the held-out questions (NOT on the 40 test questions).

The threshold only decides "obviously not in the FAQ -> say I don't know without calling the model".
The judge step handles the subtle cases, so this gate is tuned to keep nearly every answerable question
and reject the clearly unrelated ones.

  python scripts/calibrate.py            # print the table
  python scripts/calibrate.py --write    # also save config/calibrated.json
"""
import _bootstrap  # noqa: F401

import argparse
import json

from nirbhor.config import CALIBRATED_PATH, DATA_DIR, load_settings
from nirbhor.factory import build_assistant
from nirbhor.faq import load_cases
from nirbhor.glossary import expand
from nirbhor.lang import detect_language


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()

    s = load_settings(use_embeddings=False if args.no_embeddings else None)
    a = build_assistant(s, with_llm=False)
    cases = load_cases(DATA_DIR / "heldout.csv")
    scored = []
    for c in cases:
        lang = detect_language(c.question)
        top = a.retriever.search(c.question, expand(c.question) if lang != "en" else "", k=1)[0]
        scored.append((c, top.faq_id, top.score))
    n_in = sum(c.expected_faq_id is not None for c, _, _ in scored)
    n_out = len(scored) - n_in

    print(f"{'tau':>5} | in-scope kept & right | out-of-scope rejected")
    best, best_tau = -1.0, 0.0
    for i in range(0, 91, 2):
        tau = i / 100
        kept = sum(1 for c, fid, sc in scored if c.expected_faq_id is not None and fid == c.expected_faq_id and sc >= tau)
        rej = sum(1 for c, _, sc in scored if c.expected_faq_id is None and sc < tau)
        objective = kept + 0.5 * rej   # favour recall: the judge can still reject later, a wrong early refusal can't be undone
        print(f"{tau:5.2f} | {kept:>2}/{n_in}                | {rej:>2}/{n_out}")
        if objective > best:
            best, best_tau = objective, tau
    wrong_top1 = sum(1 for c, fid, _ in scored if c.expected_faq_id is not None and fid != c.expected_faq_id)
    print(f"\nrecommended retrieval_min = {best_tau:.2f}   (retrieval ranked the wrong FAQ #1 for {wrong_top1}/{n_in} in-scope questions)")
    if args.write:
        CALIBRATED_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATED_PATH.write_text(json.dumps({"retrieval_min": best_tau}, indent=2), encoding="utf-8")
        print(f"saved {CALIBRATED_PATH}")


if __name__ == "__main__":
    main()
