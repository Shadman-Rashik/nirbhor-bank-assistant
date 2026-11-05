"""Chat clients. OllamaChat runs the small model locally; OpenAICompatChat is only used to compare with an online model."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

_LOCAL_SESSION = None


def local_session():
    """HTTP session for talking to Ollama on this machine.

    It ignores HTTP(S)_PROXY / system proxy settings on purpose: a company VPN or proxy that is fine for the internet
    often breaks calls to localhost (a confusing 502 error). Ollama is local, so no proxy is wanted.
    """
    global _LOCAL_SESSION
    if _LOCAL_SESSION is None:
        import requests

        _LOCAL_SESSION = requests.Session()
        _LOCAL_SESSION.trust_env = False
    return _LOCAL_SESSION


@dataclass
class LLMResult:
    text: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient(Protocol):
    model: str

    def chat(self, messages: list[dict], max_new_tokens: int = 200, temperature: float = 0.0) -> LLMResult: ...


class OllamaChat:
    def __init__(self, base_url: str, model: str, num_ctx: int = 2048, timeout: int = 180):
        self.base_url, self.model, self.num_ctx, self.timeout = base_url.rstrip("/"), model, num_ctx, timeout

    def chat(self, messages: list[dict], max_new_tokens: int = 200, temperature: float = 0.0) -> LLMResult:
        t0 = time.perf_counter()
        r = local_session().post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": max_new_tokens,
                    "seed": 0,
                },
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return LLMResult(
            text=data["message"]["content"].strip(),
            seconds=time.perf_counter() - t0,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )


class OpenAICompatChat:
    """Any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Gemini's compatibility endpoint ...)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    def chat(self, messages: list[dict], max_new_tokens: int = 400, temperature: float = 0.0) -> LLMResult:
        import requests  # online endpoints may need the system proxy, so this one keeps the default behaviour

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_new_tokens},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"].strip(),
            seconds=time.perf_counter() - t0,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
