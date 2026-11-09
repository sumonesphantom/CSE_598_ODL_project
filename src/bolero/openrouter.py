"""Client for OpenRouter's Decisions API.

The Decisions API takes a model id, an arbitrary JSON `state`, and a
`questions` schema, and returns structured `answers`. This module adds the
operational bits the deciders need: retries with backoff, an on-disk response
cache keyed by the request, per-call cost accounting, and error redaction so an
API key can never leak into logs, cache files, or exception text.

Only the response cache is written to disk; headers (and therefore the key) are
never stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from .config import OPENROUTER_DECISIONS_URL, RESULTS_DIR

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 524, 529})
DEFAULT_CACHE_DIR = RESULTS_DIR / "deciders" / "cache"


class OpenRouterError(RuntimeError):
    """A Decisions API failure with the status and a redacted response body."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: dict | list | str | None = None,
    ) -> None:
        self.status = status
        self.message = message
        self.body = body
        super().__init__(message)


class BudgetExceeded(RuntimeError):
    """Raised when accumulated decision cost would exceed the configured cap."""


@dataclass
class DecisionResponse:
    answers: dict
    model: str
    provider: str
    cost: float
    original_cost: float
    input_tokens: int
    output_tokens: int
    cached: bool
    latency_ms: float
    raw: dict


class CostGuard:
    """Thread-safe running total of decision-model spend."""

    def __init__(self, max_cost: float | None = 2.0) -> None:
        self.max_cost = max_cost
        self.total = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            total = self.total
        if self.max_cost is not None and total >= self.max_cost:
            raise BudgetExceeded(
                f"decision-model budget exhausted: spent ${total:.6f} of ${self.max_cost:.6f}"
            )

    def add(self, cost: float) -> None:
        with self._lock:
            self.total += float(cost or 0.0)
            total = self.total
        if self.max_cost is not None and total > self.max_cost:
            raise BudgetExceeded(
                f"decision-model budget exceeded: spent ${total:.6f} of ${self.max_cost:.6f}"
            )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class DecisionsClient:
    """Synchronous client for one Decisions API endpoint, safe across threads."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        max_cost: float | None = 2.0,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff: float = 0.5,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or the environment."
            )
        self._api_key = key
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.guard = CostGuard(max_cost)
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff = backoff
        self.session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._lock = threading.Lock()

    # -- public ---------------------------------------------------------

    def ask(
        self,
        model: str,
        state: dict | list,
        questions: dict,
        use_cache: bool = True,
        session_id: str | None = None,
    ) -> DecisionResponse:
        self.guard.check()

        cache_path = self.cache_dir / f"{self._cache_key(model, state, questions)}.json"
        if use_cache:
            started = time.perf_counter()
            raw = self._read_cache(cache_path)
            if raw is not None:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return self._build(raw, model, cost=0.0, cached=True, latency_ms=elapsed_ms)

        payload: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": questions,
            "provider": {"allow_fallbacks": True},
        }
        if session_id is not None:
            payload["session_id"] = session_id

        started = time.perf_counter()
        raw = self._post(payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response = self._build(raw, model, cost=None, cached=False, latency_ms=elapsed_ms)

        if use_cache:
            self._write_cache(cache_path, model, raw)
        self.guard.add(response.cost)
        return response

    # -- request --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        }

    def _post(self, payload: dict) -> dict:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            self.guard.check()
            try:
                response = self.session.post(
                    OPENROUTER_DECISIONS_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt + 1 >= attempts:
                    raise OpenRouterError(self._redact(str(exc))) from exc
                self._sleep(self._delay(attempt, None))
                continue

            status = response.status_code
            if status == 200:
                return self._parse_success(response)

            body, message = self._error_details(response, status)
            if status in RETRYABLE_STATUS and attempt + 1 < attempts:
                self._sleep(self._delay(attempt, response.headers.get("Retry-After")))
                continue
            raise OpenRouterError(message, status=status, body=body)

        # The loop always returns or raises; this keeps type checkers quiet.
        raise OpenRouterError("request failed without a response")

    def _parse_success(self, response: Any) -> dict:
        try:
            raw = response.json()
        except ValueError:
            body = self._redact(getattr(response, "text", "") or "")
            raise OpenRouterError("response was not valid JSON", status=200, body=body) from None
        if not isinstance(raw, dict):
            raise OpenRouterError(
                "response was not a JSON object", status=200, body=self._redact(raw)
            )
        if not isinstance(raw.get("answers"), dict):
            raise OpenRouterError(
                "response is missing an 'answers' object",
                status=200,
                body=self._redact(raw),
            )
        return raw

    def _error_details(self, response: Any, status: int) -> tuple[dict | list | str | None, str]:
        """Return (redacted body, redacted message) for an error response."""
        parsed: Any = None
        try:
            parsed = response.json()
        except ValueError:
            parsed = None

        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict) and error.get("message"):
                message = self._redact(str(error["message"]))
            else:
                message = f"HTTP {status}"
            return self._redact(parsed), message

        text = getattr(response, "text", "") or ""
        message = self._redact(text) if text else f"HTTP {status}"
        return self._redact(text), message

    def _delay(self, attempt: int, retry_after: Any) -> float:
        if retry_after is not None:
            try:
                return float(int(retry_after))
            except (TypeError, ValueError):
                pass
        return float(self.backoff * (2**attempt))

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    # -- response mapping -----------------------------------------------

    def _build(
        self,
        raw: dict,
        requested_model: str,
        cost: float | None,
        cached: bool,
        latency_ms: float,
    ) -> DecisionResponse:
        usage = raw.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        original_cost = _as_float(usage.get("cost"))
        provider = raw.get("provider")
        if not isinstance(provider, str):
            provider = "" if provider is None else str(provider)
        return DecisionResponse(
            answers=dict(raw["answers"]),
            model=raw.get("model") or requested_model,
            provider=provider,
            cost=original_cost if cost is None else cost,
            original_cost=original_cost,
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cached=cached,
            latency_ms=latency_ms,
            raw=raw,
        )

    # -- cache ----------------------------------------------------------

    @staticmethod
    def _cache_key(model: str, state: dict | list, questions: dict) -> str:
        canonical = json.dumps(
            {"model": model, "state": state, "questions": questions},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _read_cache(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        raw = data.get("response")
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            return None
        return raw

    def _write_cache(self, path: Path, model: str, raw: dict) -> None:
        # Headers are deliberately absent: the key must never reach disk.
        payload = json.dumps({"model": model, "response": raw})
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload)
            except OSError:
                # A cache failure must never fail an otherwise good decision.
                pass
