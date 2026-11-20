import asyncio
import json

import httpx
import pytest

from hidagent.llm import LLM


def fake_server(llm: LLM, wants: str) -> list[dict]:
    """Point the client at an in-memory endpoint that only accepts the `wants` token parameter."""
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if wants not in body:
            return httpx.Response(400, json={"error": {"message": f"Unsupported parameter. Use '{wants}' instead."}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "DO: wait"}}]})

    llm._http = httpx.AsyncClient(base_url="http://test/v1", transport=httpx.MockTransport(handle))
    return seen


def test_classic_servers_get_max_tokens_once():
    llm = LLM("http://test/v1", "qwen")
    seen = fake_server(llm, "max_tokens")
    assert asyncio.run(llm.chat("sys", "hi")) == "DO: wait"
    assert len(seen) == 1 and seen[0]["max_tokens"] == 2000


def test_auto_switches_to_max_completion_tokens_and_remembers():
    llm = LLM("http://test/v1", "gpt")
    seen = fake_server(llm, "max_completion_tokens")
    asyncio.run(llm.chat("sys", "hi"))
    asyncio.run(llm.chat("sys", "again"))
    assert [("max_tokens" in b, "max_completion_tokens" in b) for b in seen] == [(True, False), (False, True),
                                                                                 (False, True)]


def test_explicit_token_param_is_used_directly():
    llm = LLM("http://test/v1", "gpt", token_param="max_completion_tokens")
    seen = fake_server(llm, "max_completion_tokens")
    asyncio.run(llm.chat("sys", "hi"))
    assert len(seen) == 1


def test_extra_body_and_image_are_sent():
    llm = LLM("http://test/v1", "gpt", extra={"reasoning_effort": "none"})
    seen = fake_server(llm, "max_tokens")
    asyncio.run(llm.chat("sys", "hi", image=b"\xff\xd8jpeg"))
    body = seen[0]
    assert body["reasoning_effort"] == "none"
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_reasoning_that_runs_out_is_retried_with_more_room():
    # deepseek-flash, 2026-11-17: thinking used all 600 tokens, content came back empty ("actor gave no valid action")
    from hidagent.llm import cost
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append(body["max_tokens"])
        usage = {"prompt_tokens": 100, "completion_tokens": body["max_tokens"] // 2, "prompt_cache_hit_tokens": 40,
                 "completion_tokens_details": {"reasoning_tokens": 50}}
        if len(seen) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                                             "usage": usage})
        return httpx.Response(200, json={"choices": [{"message": {"content": "DO: wait"}, "finish_reason": "stop"}],
                                         "usage": usage})

    llm = LLM("http://test/v1", "deepseek-flash")
    llm._http = httpx.AsyncClient(base_url="http://test/v1", transport=httpx.MockTransport(handle))
    assert asyncio.run(llm.chat("sys", "hi", max_tokens=600)) == "DO: wait"
    assert seen == [600, 1200]
    assert llm.last_usage == {"prompt": 200, "completion": 900, "reasoning": 100, "cache_hit": 80}
    assert cost(llm.last_usage, "1,2,3") == round((80 * 1 + 120 * 2 + 900 * 3) / 1e6, 6)
    assert cost(llm.last_usage, "") is None


def test_price_follows_deepseek_peak_hours_in_beijing_time():
    from datetime import datetime

    from hidagent.llm import BEIJING, price_at

    def at(y, mo, d, h, mi=0):
        return datetime(y, mo, d, h, mi, tzinfo=BEIJING).timestamp()

    peak, idle = "0.04,2,8", "0.02,1,4"
    assert price_at(at(2026, 11, 17, 9, 0), peak, idle) == peak       # Tuesday 9:00
    assert price_at(at(2026, 11, 17, 11, 59), peak, idle) == peak
    assert price_at(at(2026, 11, 17, 12, 30), peak, idle) == idle     # lunch break
    assert price_at(at(2026, 11, 17, 17, 59), peak, idle) == peak
    assert price_at(at(2026, 11, 17, 18, 0), peak, idle) == idle
    assert price_at(at(2026, 11, 17, 23, 0), peak, idle) == idle
    assert price_at(at(2026, 11, 21, 10, 0), peak, idle) == idle      # Saturday
    assert price_at(at(2026, 11, 17, 23, 0), peak) == peak            # no idle price set: always peak


def test_fast_calls_add_the_fast_body_and_careful_ones_do_not():
    llm = LLM("http://test/v1", "deepseek", extra={"a": 1}, fast_extra={"thinking": {"type": "disabled"}})
    seen = fake_server(llm, "max_tokens")
    asyncio.run(llm.chat("sys", "hi", fast=True))
    asyncio.run(llm.chat("sys", "hi"))
    assert seen[0]["thinking"] == {"type": "disabled"} and seen[0]["a"] == 1
    assert "thinking" not in seen[1] and seen[1]["a"] == 1


def _flaky(statuses, monkeypatch):
    """An LLM whose server answers with these status codes in turn (then 200), without real waiting."""
    from hidagent import llm as llm_module

    monkeypatch.setattr(llm_module, "RETRY_WAITS", (0.0, 0.0))
    calls = []

    def handle(request):
        calls.append(1)
        code = statuses[len(calls) - 1] if len(calls) <= len(statuses) else 200
        if code != 200:
            return httpx.Response(code, text="busy")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    llm = LLM("http://test/v1", "m")
    llm._http = httpx.AsyncClient(base_url="http://test/v1", transport=httpx.MockTransport(handle))
    return llm, calls


def test_busy_api_is_retried(monkeypatch):
    llm, calls = _flaky([503, 429], monkeypatch)
    assert asyncio.run(llm.chat("s", "t")) == "ok" and len(calls) == 3


def test_retries_give_up_after_three_tries(monkeypatch):
    llm, calls = _flaky([503, 503, 503, 503], monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(llm.chat("s", "t"))
    assert len(calls) == 3


def test_a_bad_request_is_not_retried(monkeypatch):
    llm, calls = _flaky([401], monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(llm.chat("s", "t"))
    assert len(calls) == 1
