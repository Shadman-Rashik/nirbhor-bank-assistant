"""Type questions in your terminal. Shows what happened at each step.  python scripts/chat_cli.py [--no-embeddings]"""
import _bootstrap  # noqa: F401

import argparse

from nirbhor.config import load_settings
from nirbhor.factory import build_assistant

p = argparse.ArgumentParser()
p.add_argument("--gen-model")
p.add_argument("--no-embeddings", action="store_true")
p.add_argument("--no-judge", action="store_true")
p.add_argument("--judge", action="store_true", help="turn the strict yes/no check on (off by default)")
args = p.parse_args()

s = load_settings(gen_model=args.gen_model, use_embeddings=False if args.no_embeddings else None,
                  use_judge=False if args.no_judge else (True if args.judge else None))
bot = build_assistant(s)
print(f"model={s.gen_model}  embeddings={'off' if not s.use_embeddings else s.embed_model}  (empty line to quit)")
bot.warmup()
while True:
    q = input("\nyou> ").strip()
    if not q:
        break
    a = bot.answer(q)
    print(f"bot> {a.text}")
    print(f"     [{a.lang} | {a.status}: {a.reason} | FAQ {a.faq_ids or '-'} | {a.total_ms:.0f} ms | top hits {[(h['faq_id'], h['score']) for h in a.hits]}]")
