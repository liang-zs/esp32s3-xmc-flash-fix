"""Minimal client for any OpenAI-compatible chat endpoint (vision models)."""

import base64
import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
MAX_TOKENS = 2000  # reply budget incl. a reasoning model's thinking (deepseek-flash thinks ~400+ tokens per step)


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", extra: dict | None = None,
                 token_param: str = "auto"):
        self.model, self.extra = model, extra or {}
        self.last_usage: dict = {}  # tokens used by the last chat() call
        # "auto": send max_tokens; switch for good if the server asks for max_completion_tokens (newer OpenAI models)
        self.token_param, self._auto = (token_param, False) if token_param in TOKEN_PARAMS else ("max_tokens", True)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=120)

    @classmethod
    def from_env(cls, role: str) -> "LLM":
        """ROLE_* variables (e.g. ACTOR_MODEL) override the shared LLM_* ones."""
        def env(key, default=""):
            return os.environ.get(f"{role}_{key}") or os.environ.get(f"LLM_{key}", default)
        return cls(env("BASE_URL", "http://localhost:8000/v1"), env("MODEL"),
                   env("API_KEY"), json.loads(env("EXTRA_BODY", "{}")), env("TOKEN_PARAM", "auto"))

    async def chat(self, system: str, text: str, image: bytes | None = None,
                   max_tokens: int = MAX_TOKENS, temperature: float = 0) -> str:
        """The reply text. Reasoning models count their thinking in max_tokens: when it runs out before any
        answer (empty content, finish_reason "length"), ask once more with twice the room.
        Token use of the call, retries included, is left in `last_usage`."""
        content = [{"type": "text", "text": text}]
        if image:
            url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
            content.insert(0, {"type": "image_url", "image_url": {"url": url}})
        body = {"model": self.model, "temperature": temperature, **self.extra,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        self.last_usage = {}
        for limit in (max_tokens, max_tokens * 2):
            j = await self._post(body, limit)
            _add(self.last_usage, j.get("usage") or {})
            choice = j["choices"][0]
            reply = choice["message"].get("content") or ""
            if reply.strip() or choice.get("finish_reason") != "length":
                return reply
            log.warning("%s used all %d tokens thinking, no answer; retrying with %d", self.model, limit, limit * 2)
        return reply

    async def _post(self, body: dict, max_tokens: int) -> dict:
        r = await self._http.post("/chat/completions", json={self.token_param: max_tokens, **body})
        if r.status_code == 400 and self._auto and "max_completion_tokens" in r.text:
            self.token_param, self._auto = "max_completion_tokens", False
            r = await self._http.post("/chat/completions", json={self.token_param: max_tokens, **body})
        r.raise_for_status()
        return r.json()


USAGE_KEYS = ("prompt", "completion", "reasoning", "cache_hit")


def _add(total: dict, usage: dict) -> None:
    """Sum one API usage block into `total` (OpenAI and DeepSeek field names)."""
    got = {"prompt": usage.get("prompt_tokens", 0), "completion": usage.get("completion_tokens", 0),
           "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
           "cache_hit": usage.get("prompt_cache_hit_tokens",
                                  (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))}
    for k in USAGE_KEYS:
        total[k] = total.get(k, 0) + (got[k] or 0)


def cost(usage: dict, price: str) -> float | None:
    """Money spent, from "cache_hit_input,cache_miss_input,output" prices per million tokens (env LLM_PRICE)."""
    try:
        hit, miss, out = (float(v) for v in price.split(","))
    except ValueError:
        return None
    cached = usage.get("cache_hit", 0)
    return round((cached * hit + (usage.get("prompt", 0) - cached) * miss + usage.get("completion", 0) * out) / 1e6, 6)
