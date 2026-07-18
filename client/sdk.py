"""Reliable Python SDK for the deployed Document Analyst."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx

_RETRYABLE = {429, 503}


class AnalystClientError(Exception):
    """A Databricks endpoint error with transport metadata preserved."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        details = message
        if status_code is not None:
            details = f"HTTP {status_code}: {details}"
        if request_id:
            details = f"{details} (request ID: {request_id})"
        super().__init__(details)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


def _extract_answer(payload: Any) -> str:
    """Parse raw LangGraph state as well as chat-native endpoint responses."""
    if isinstance(payload, list):
        if not payload:
            raise ValueError("endpoint returned an empty result list")
        return _extract_answer(payload[0])
    if not isinstance(payload, dict):
        if isinstance(payload, str) and payload:
            return payload
        raise ValueError("endpoint returned an unsupported response shape")

    for wrapper in ("predictions", "outputs", "output"):
        if wrapper in payload:
            return _extract_answer(payload[wrapper])

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"])
            if choice.get("text"):
                return str(choice["text"])

    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        message = messages[-1]
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message["content"])
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)

    if payload.get("content") is not None:
        return str(payload["content"])
    raise ValueError("endpoint response did not contain an answer")


def _stream_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict) and delta.get("content") is not None:
                return str(delta["content"])
    try:
        return _extract_answer(payload)
    except ValueError:
        return ""


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name.strip()
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not self.endpoint_name:
            raise ValueError("endpoint_name must not be empty")
        if not self.host:
            raise ValueError("DATABRICKS_HOST is required (argument or environment)")
        if not self.token:
            raise ValueError("DATABRICKS_TOKEN is required (argument or environment)")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._headers = {"Authorization": f"Bearer {self.token}"}
        self._client = httpx.Client(timeout=self.timeout, headers=self._headers)
        self._invocations_url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        self._status_url = f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(2**attempt)

    @staticmethod
    def _error_from_response(response: httpx.Response) -> AnalystClientError:
        request_id = response.headers.get("x-databricks-request-id") or response.headers.get(
            "x-request-id"
        )
        try:
            body = response.json()
            message = body.get("message") or body.get("error") or response.text
        except (ValueError, AttributeError):
            message = response.text
        return AnalystClientError(
            str(message or response.reason_phrase),
            status_code=response.status_code,
            request_id=request_id,
        )

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        started = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - started
                raise TimeoutError(
                    f"Document Analyst request timed out after {elapsed:.3f} seconds"
                ) from exc
            except httpx.HTTPError as exc:
                raise AnalystClientError(f"Request failed: {exc}") from exc

            if response.status_code in _RETRYABLE and attempt < self.max_retries:
                self._backoff(attempt)
                continue
            if response.is_error:
                raise self._error_from_response(response)
            return response
        raise AssertionError("retry loop exhausted unexpectedly")

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        response = self._request(
            "POST",
            self._invocations_url,
            json={"messages": [{"role": "user", "content": question}]},
        )
        try:
            return _extract_answer(response.json())
        except (ValueError, json.JSONDecodeError) as exc:
            request_id = response.headers.get("x-databricks-request-id") or response.headers.get(
                "x-request-id"
            )
            raise AnalystClientError(
                f"Could not parse endpoint response: {exc}",
                status_code=response.status_code,
                request_id=request_id,
            ) from exc

    def ask_streaming(self, question: str) -> Iterator[str]:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = time.monotonic()

        for attempt in range(self.max_retries + 1):
            try:
                with self._client.stream(
                    "POST",
                    self._invocations_url,
                    headers={"Accept": "text/event-stream"},
                    json={
                        "messages": [{"role": "user", "content": question}],
                        "stream": True,
                    },
                ) as response:
                    if response.status_code in _RETRYABLE and attempt < self.max_retries:
                        response.read()
                        self._backoff(attempt)
                        continue
                    if response.is_error:
                        response.read()
                        try:
                            error_payload = response.json()
                            error_message = str(
                                error_payload.get("message")
                                or error_payload.get("error")
                                or response.text
                            )
                        except (ValueError, AttributeError):
                            error_message = response.text
                        if (
                            response.status_code == 400
                            and "does not support streaming" in error_message.lower()
                        ):
                            answer = self.ask(question)
                            if answer:
                                yield answer
                            return
                        raise self._error_from_response(response)

                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" not in content_type:
                        response.read()
                        try:
                            answer = _extract_answer(response.json())
                        except ValueError as exc:
                            raise AnalystClientError(
                                f"Could not parse endpoint response: {exc}",
                                status_code=response.status_code,
                            ) from exc
                        if answer:
                            yield answer
                        return

                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        data = line[5:].strip() if line.startswith("data:") else line
                        if data == "[DONE]":
                            return
                        try:
                            chunk = _stream_text(json.loads(data))
                        except json.JSONDecodeError:
                            chunk = data
                        if chunk:
                            yield chunk
                    return
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - started
                raise TimeoutError(
                    f"Document Analyst stream timed out after {elapsed:.3f} seconds"
                ) from exc
            except httpx.HTTPError as exc:
                raise AnalystClientError(f"Streaming request failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            response = self._request("GET", self._status_url)
            payload = response.json()
        except (AnalystClientError, TimeoutError, ValueError):
            return False
        state = payload.get("state", {}) if isinstance(payload, dict) else {}
        ready = state.get("ready") if isinstance(state, dict) else None
        return str(ready).upper() == "READY"
