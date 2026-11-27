"""Offline tests for the OpenRouter Decisions API client.

No network access: a small FakeSession stands in for requests.Session, and the
client's injected `sleep` keeps retries instant.
"""

from __future__ import annotations

import json

import pytest

from bolero.openrouter import (
    BudgetExceeded,
    CostGuard,
    DecisionsClient,
    OpenRouterError,
)

KEY = "sk-test-secret-value"
MODEL = "typesafe/jev-1.14"
STATE = {"evidence": {"image": {"width": 128, "height": 96}}}
QUESTIONS = {"degraded": {"type": "noul"}}


class FakeResponse:
    def __init__(self, status_code=200, data=None, headers=None, text=None):
        self.status_code = status_code
        self._data = data
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(data) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("no json body")
        return self._data


class FakeSession:
    """Returns queued responses in order and records every POST."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        self.headers.append(dict(headers or {}))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def success(data=None, cost=0.000123, provider="TypeSafe"):
    data = data if data is not None else {
        "answers": {"degraded": True, "condition": "blur", "severity": "mild"},
        "model": MODEL,
        "provider": provider,
        "usage": {"cost": cost, "input_tokens": 321, "output_tokens": 12},
    }
    return FakeResponse(200, data)


def client(tmp_path, responses, **kwargs):
    sleep_calls = []
    kwargs.setdefault("session", FakeSession(responses))
    kwargs.setdefault("sleep", sleep_calls.append)
    kwargs.setdefault("cache_dir", tmp_path / "cache")
    kwargs.setdefault("api_key", KEY)
    return DecisionsClient(**kwargs), kwargs["session"], sleep_calls


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        DecisionsClient()
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


def test_success_payload_and_mapping(tmp_path):
    c, session, _ = client(tmp_path, [success()])
    response = c.ask(MODEL, STATE, QUESTIONS)

    assert response.answers == {"degraded": True, "condition": "blur", "severity": "mild"}
    assert response.model == MODEL
    assert response.provider == "TypeSafe"
    assert response.cost == pytest.approx(0.000123)
    assert response.original_cost == pytest.approx(0.000123)
    assert response.input_tokens == 321
    assert response.output_tokens == 12
    assert response.cached is False
    assert response.raw["provider"] == "TypeSafe"

    assert len(session.calls) == 1
    headers = session.headers[0]
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert headers["Content-Type"] == "application/json"
    assert headers["TypeSafe-Danger-Zone-Images"] == "true"

    payload = session.calls[0]["json"]
    assert payload["model"] == MODEL
    assert payload["state"] == STATE
    assert payload["questions"] == QUESTIONS
    assert payload["provider"] == {"allow_fallbacks": True}


def test_cache_hit_skips_network(tmp_path):
    c, session, _ = client(tmp_path, [success()])
    first = c.ask(MODEL, STATE, QUESTIONS)
    second = c.ask(MODEL, STATE, QUESTIONS)

    assert len(session.calls) == 1
    assert first.cached is False
    assert second.cached is True
    assert second.cost == 0.0
    assert second.original_cost == pytest.approx(0.000123)
    assert second.answers == first.answers
    assert second.latency_ms >= 0.0

    cache_file = next((tmp_path / "cache").glob("*.json"))
    assert KEY not in cache_file.read_text()


def test_corrupt_cache_is_a_miss(tmp_path):
    c, session, _ = client(tmp_path, [success(), success()])
    c.ask(MODEL, STATE, QUESTIONS)
    cache_file = next((tmp_path / "cache").glob("*.json"))
    cache_file.write_text("not json")

    second = c.ask(MODEL, STATE, QUESTIONS)
    assert len(session.calls) == 2
    assert second.cached is False


def test_cache_key_ignores_dict_order(tmp_path):
    c, session, _ = client(tmp_path, [success()])
    c.ask(MODEL, {"a": 1, "b": {"x": 2, "y": 3}}, QUESTIONS)
    second = c.ask(MODEL, {"b": {"y": 3, "x": 2}, "a": 1}, QUESTIONS)

    assert len(session.calls) == 1
    assert second.cached is True


def test_retry_on_429_then_success(tmp_path):
    c, session, sleeps = client(
        tmp_path,
        [FakeResponse(429, {"error": {"message": "slow down"}}, headers={"Retry-After": "1"}), success()],
    )
    response = c.ask(MODEL, STATE, QUESTIONS)

    assert len(session.calls) == 2
    assert sleeps == [1.0]
    assert response.cached is False


def test_retry_exhausted_raises(tmp_path):
    c, session, _ = client(tmp_path, [FakeResponse(503, {"error": {"message": "unavailable"}})], max_retries=2)

    with pytest.raises(OpenRouterError) as excinfo:
        c.ask(MODEL, STATE, QUESTIONS)

    assert excinfo.value.status == 503
    assert len(session.calls) == 2 + 1


def test_auth_error_is_not_retried(tmp_path):
    c, session, _ = client(
        tmp_path,
        [FakeResponse(401, {"error": {"message": f"invalid key {KEY}"}})],
    )

    with pytest.raises(OpenRouterError) as excinfo:
        c.ask(MODEL, STATE, QUESTIONS)

    assert len(session.calls) == 1
    assert excinfo.value.status == 401
    assert KEY not in str(excinfo.value)
    assert KEY not in excinfo.value.message


def test_error_body_is_redacted(tmp_path):
    c, _, _ = client(tmp_path, [FakeResponse(400, None, text=f"bad request for {KEY}")])

    with pytest.raises(OpenRouterError) as excinfo:
        c.ask(MODEL, STATE, QUESTIONS)

    error = excinfo.value
    assert KEY not in str(error)
    assert KEY not in str(error.body)
    assert KEY not in error.message


def test_cost_guard_trips_after_response(tmp_path):
    c, session, _ = client(tmp_path, [success(cost=0.01)], max_cost=1e-9)

    with pytest.raises(BudgetExceeded):
        c.ask(MODEL, STATE, QUESTIONS)
    assert len(session.calls) == 1

    with pytest.raises(BudgetExceeded):
        c.ask(MODEL, STATE, QUESTIONS)
    assert len(session.calls) == 1


def test_missing_answers_raises(tmp_path):
    c, _, _ = client(tmp_path, [FakeResponse(200, {"model": MODEL, "usage": {"cost": 0.0}})])

    with pytest.raises(OpenRouterError):
        c.ask(MODEL, STATE, QUESTIONS)


def test_cost_guard_unit():
    guard = CostGuard(max_cost=1.0)
    guard.check()
    guard.add(0.5)
    assert guard.total == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        guard.add(0.75)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_no_cost_limit_allows_anything():
    guard = CostGuard(max_cost=None)
    guard.check()
    guard.add(100.0)
    guard.check()
    assert guard.total == pytest.approx(100.0)
