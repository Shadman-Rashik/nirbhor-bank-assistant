"""Start the web UI.  python scripts/serve_ui.py   then open http://127.0.0.1:8000

  python scripts/serve_ui.py --gen-model gemma3:4b
  python scripts/serve_ui.py --retrieval-only --no-embeddings     # look at the UI without any model installed
Nothing else to install: the UI uses only the Python standard library.
"""
import _bootstrap  # noqa: F401

import argparse
import sys
import webbrowser

from nirbhor.config import ROOT, load_settings
from nirbhor.factory import build_assistant
from nirbhor.webapp import WebApp, make_server

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--gen-model")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--no-embeddings", action="store_true")
p.add_argument("--no-judge", action="store_true")
p.add_argument("--judge", action="store_true", help="turn the strict yes/no check on (off by default)")
p.add_argument("--retrieval-only", action="store_true", help="no generation model: replies are the FAQ text itself")
p.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
args = p.parse_args()

s = load_settings(gen_model=args.gen_model, use_embeddings=False if args.no_embeddings else None,
                  use_judge=False if args.no_judge else (True if args.judge else None))
print("Loading the model (the first start can take a minute) ...", flush=True)
try:
    bot = build_assistant(s, with_llm=not args.retrieval_only)
    bot.warmup()
except Exception as e:  # noqa: BLE001
    sys.exit(f"\nCannot reach the model server at {s.ollama_url}: {type(e).__name__}: {e}\n"
             f"Is Ollama running? Did you run:  ollama pull {s.gen_model}   and   ollama pull {s.embed_model}  ?\n"
             "(To look at the UI without a model: add --retrieval-only --no-embeddings)")

app = WebApp(bot, None if args.retrieval_only else s.gen_model, ROOT / "logs")
server = make_server(app, "127.0.0.1", args.port)
url = f"http://127.0.0.1:{args.port}"
print(f"\nReady:  {url}   (press Ctrl+C to stop)\nHand-over requests are saved in logs/handover_requests.jsonl", flush=True)
if not args.no_open:
    webbrowser.open(url)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nStopped.")
