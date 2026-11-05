"""Builds a ready-to-use Assistant from Settings (shared by the CLI, the eval scripts and the Streamlit app)."""
from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR, RESULTS_DIR, Settings
from .faq import load_faq
from .llm import OllamaChat
from .pipeline import Assistant
from .retriever import FaqRetriever, OllamaEmbedder


def build_assistant(
    settings: Settings,
    faq_path: str | Path = DATA_DIR / "bank-faq.csv",
    with_llm: bool = True,
    cache_path: str | Path | None = None,
) -> Assistant:
    faq = load_faq(faq_path)
    embedder = None
    if settings.use_embeddings:
        embedder = OllamaEmbedder(
            settings.ollama_url, settings.embed_model,
            cache_path=Path(cache_path) if cache_path else RESULTS_DIR / ".embed_cache.json",
            timeout=settings.request_timeout_s,
        )
    retriever = FaqRetriever(faq, embedder, settings.w_dense)
    llm = OllamaChat(settings.ollama_url, settings.gen_model, settings.num_ctx, settings.request_timeout_s) if with_llm else None
    return Assistant(faq, retriever, llm, settings)
