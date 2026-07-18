"""Offline tests for response parsing, health checks, retries, and errors."""

from __future__ import annotations

import httpx
import pytest

from client.sdk import AnalystClientError, DocumentAnalystClient


def _client(handler, *, max_retries=0):
    client = DocumentAnalystClient(
        "analyst", host="https://workspace.example", token="token", max_retries=max_retries
    )
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=1)
    client._backoff = lambda _attempt: None
    return client


def test_ask_parses_raw_langgraph_state():
    def handler(request):
        assert request.url.path.endswith("/invocations")
        return httpx.Response(
            200,
            json=[{"messages": [{"role": "assistant", "content": "cited answer"}]}],
        )

    with _client(handler) as client:
        assert client.ask("question") == "cited answer"


def test_health_check_only_accepts_ready():
    with _client(
        lambda _request: httpx.Response(200, json={"state": {"ready": "READY"}})
    ) as client:
        assert client.health_check() is True

    with _client(
        lambda _request: httpx.Response(200, json={"state": {"ready": "NOT_READY"}})
    ) as client:
        assert client.health_check() is False


def test_retry_then_success():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, json={"message": "scaling"})
        return httpx.Response(
            200,
            json=[{"messages": [{"role": "assistant", "content": "ready"}]}],
        )

    with _client(handler, max_retries=2) as client:
        assert client.ask("question") == "ready"
    assert calls == 3


def test_http_error_preserves_metadata():
    response_headers = {"x-databricks-request-id": "request-123"}
    with _client(
        lambda _request: httpx.Response(
            400, json={"message": "bad input"}, headers=response_headers
        )
    ) as client:
        with pytest.raises(AnalystClientError) as error:
            client.ask("question")
    assert error.value.status_code == 400
    assert error.value.request_id == "request-123"


def test_streaming_accepts_single_non_incremental_response():
    payload = [{"messages": [{"role": "assistant", "content": "one complete answer"}]}]
    with _client(lambda _request: httpx.Response(200, json=payload)) as client:
        assert list(client.ask_streaming("question")) == ["one complete answer"]


def test_streaming_falls_back_when_endpoint_rejects_streaming():
    calls = []

    def handler(request):
        calls.append(request)
        payload = request.read()
        if b'"stream":true' in payload.replace(b" ", b""):
            return httpx.Response(400, json={"message": "This endpoint does not support streaming."})
        return httpx.Response(
            200,
            json=[{"messages": [{"role": "assistant", "content": "fallback answer"}]}],
        )

    with _client(handler) as client:
        assert list(client.ask_streaming("question")) == ["fallback answer"]
    assert len(calls) == 2


def test_streaming_parses_sse_deltas():
    stream = (
        'data: {"choices":[{"delta":{"content":"first "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"second"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    with _client(
        lambda _request: httpx.Response(
            200,
            text=stream,
            headers={"content-type": "text/event-stream"},
        )
    ) as client:
        assert list(client.ask_streaming("question")) == ["first ", "second"]
