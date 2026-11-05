"""Settings: defaults < config/calibrated.json < NIRBHOR_* environment variables < explicit overrides."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CALIBRATED_PATH = ROOT / "config" / "calibrated.json"


@dataclass
class Settings:
    # --- models (served by Ollama on this machine, no data leaves it) ---
    ollama_url: str = "http://localhost:11434"
    gen_model: str = "gemma3:4b"         # chosen on the held-out set (see README, "How the final configuration was chosen")
    embed_model: str = "bge-m3"
    # --- pipeline switches ---
    use_embeddings: bool = True          # False = lexical-only retrieval (no embedding model needed)
    use_judge: bool = False               # second, strict "does this FAQ answer the question?" check
    judge_examples: bool = False         # add a few worked examples to the judge prompt (A/B it with --judge-examples)
    use_query_translation: bool = False  # ask the small model to translate bn/banglish queries first
    # --- retrieval ---
    retrieval_min: float = 0.30          # below this top-1 score: say "I don't know" without calling the LLM
    w_dense: float = 0.6                 # weight of embedding similarity (rest = keyword score)
    top_k: int = 3
    # --- generation ---
    temperature: float = 0.0
    num_ctx: int = 2048
    max_new_tokens_en: int = 160
    max_new_tokens_banglish: int = 220
    max_new_tokens_bn: int = 400         # Bangla script costs many more tokens per word
    request_timeout_s: int = 180


def load_settings(**overrides) -> Settings:
    values: dict = {}
    if CALIBRATED_PATH.exists():
        values.update(json.loads(CALIBRATED_PATH.read_text(encoding="utf-8")))
    for f in fields(Settings):
        env = os.environ.get(f"NIRBHOR_{f.name.upper()}")
        if env is None:
            continue
        if f.type in ("bool", bool):
            values[f.name] = env.strip().lower() in ("1", "true", "yes", "on")
        elif f.type in ("int", int):
            values[f.name] = int(env)
        elif f.type in ("float", float):
            values[f.name] = float(env)
        else:
            values[f.name] = env
    values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in values.items() if k in known})
