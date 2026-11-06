"""Offline checks for ``player_pool.llm_client``.

Runs every request against a local mock OpenCode Console server, so no API key
and no network access is needed:

    python -m player_pool.check_llm_client
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from player_pool.llm_client import (
    API_KEY_ENVS,
    DEFAULT_MODEL,
    ENV_FILE_ENV,
    LLMClient,
    LLMError,
    MissingAPIKeyError,
)

Responder = Callable[[Dict[str, Any]], Tuple[int, Dict[str, Any], Dict[str, str]]]


def _header(record: Dict[str, Any], name: str) -> Optional[str]:
    lowered = {key.lower(): value for key, value in record["headers"].items()}
    return lowered.get(name.lower())


def _chat_payload(
    text: str = "hi from go",
    opencode: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    usage: Dict[str, Any] = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "cost": 0.001,
    }
    if opencode is not None:
        usage["opencode"] = opencode
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _start_server(responder: Responder):
    calls: List[Dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._handle()

        def do_POST(self) -> None:  # noqa: N802
            self._handle()

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            record = {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(body) if body else None,
            }
            calls.append(record)
            status, payload, extra_headers = responder(record)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in extra_headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # silence request log
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", calls


def _client(base_url: str, **kwargs: Any) -> LLMClient:
    defaults = {
        "api_key": "oc_sk_test",
        "session_id": "ses_test00000000000000000000000000",
        "timeout": 5.0,
        "max_retries": 0,
    }
    defaults.update(kwargs)
    return LLMClient(model=DEFAULT_MODEL, base_url=base_url, **defaults)


def _pop_key_envs() -> Dict[str, str]:
    saved = {}
    for name in API_KEY_ENVS:
        value = os.environ.pop(name, None)
        if value is not None:
            saved[name] = value
    return saved


def check_default_urls() -> None:
    client = LLMClient(api_key="x")
    assert client.base_url == "https://opencode.ai/inference/go", client.base_url
    assert (
        client.chat_url
        == "https://opencode.ai/inference/go/openai/v1/chat/completions"
    ), client.chat_url
    assert (
        client.models_url == "https://opencode.ai/inference/go/v1/models"
    ), client.models_url
    assert LLMClient(provider="opencode", api_key="x").base_url == (
        "https://opencode.ai/inference"
    )
    print("ok  default OpenCode Go URLs")


def check_chat_request_and_usage() -> None:
    opencode = {
        "billing_source": "go",
        "allowance": 0.04,
        "list_cost": 0.001,
        "tokens": {"input": 12, "output": 3},
    }
    responder = lambda record: (200, _chat_payload(opencode=opencode), {})
    server, base, calls = _start_server(responder)
    try:
        client = _client(base)
        response = client.chat(
            "hello",
            system="You are brief.",
            temperature=0.2,
            max_tokens=64,
        )
        assert response.text == "hi from go", response.text
        assert response.usage.prompt_tokens == 12
        assert response.usage.cost == 0.001
        assert response.usage.allowance == 0.04, response.usage
        assert response.usage.billing_source == "go"
        assert response.model == DEFAULT_MODEL
        assert response.finish_reason == "stop"

        record = calls[0]
        assert record["method"] == "POST"
        assert record["path"] == "/openai/v1/chat/completions", record["path"]
        assert _header(record, "Authorization") == "Bearer oc_sk_test"
        session = _header(record, "x-opencode-session")
        assert session == "ses_test00000000000000000000000000", session
        assert (_header(record, "User-Agent") or "").startswith("metamon-player-pool/")
        body = record["body"]
        assert body["model"] == DEFAULT_MODEL
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 64
        assert body["messages"][0] == {"role": "system", "content": "You are brief."}
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
    finally:
        server.shutdown()
    print("ok  chat request shape + usage/opencode parsing")


def check_string_prompt_and_message_list() -> None:
    server, base, calls = _start_server(
        lambda record: (200, _chat_payload(), {})
    )
    try:
        client = _client(base)
        client.chat(
            [
                {"role": "system", "content": "existing"},
                {"role": "user", "content": "move?"},
            ],
            system="prepended",
        )
        body = calls[0]["body"]
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "prepended\n\nexisting"
        assert body["messages"][1] == {"role": "user", "content": "move?"}
    finally:
        server.shutdown()
    print("ok  system prompt merging + message list passthrough")


def check_retry_on_429() -> None:
    attempts = {"n": 0}

    def responder(record: Dict[str, Any]):
        if attempts["n"] == 0:
            attempts["n"] += 1
            return (
                429,
                {"error": {"type": "rate_limit", "message": "slow down"}},
                {"Retry-After": "0"},
            )
        return 200, _chat_payload("after retry"), {}

    server, base, calls = _start_server(responder)
    try:
        client = _client(base, max_retries=2)
        response = client.chat("hello")
        assert response.text == "after retry"
        assert len(calls) == 2, len(calls)
    finally:
        server.shutdown()
    print("ok  429 retried with Retry-After")


def check_http_error_mapping() -> None:
    server, base, calls = _start_server(
        lambda record: (
            401,
            {"error": {"type": "server_error", "message": "Go requires a credential"}},
            {},
        )
    )
    try:
        client = _client(base, api_key="wrong")
        try:
            client.chat("hello")
        except LLMError as error:
            assert error.status == 401, error.status
            assert error.error_type == "server_error"
            assert "credential" in str(error)
        else:
            raise AssertionError("401 did not raise")
        assert len(calls) == 1, "401 must not be retried"
    finally:
        server.shutdown()
    print("ok  HTTP error becomes LLMError(status, type)")


def check_missing_api_key() -> None:
    # point the .env loader at an empty file so a repo .env cannot inject a key
    blocker = Path("/tmp/empty-player-pool-env")
    blocker.write_text("")
    saved = _pop_key_envs()
    prior_env_file = os.environ.get(ENV_FILE_ENV)
    os.environ[ENV_FILE_ENV] = str(blocker)
    try:
        client = LLMClient()  # no key anywhere
        try:
            client.chat("hello")
        except MissingAPIKeyError as error:
            assert "OPENCODE_API_KEY" in str(error)
        else:
            raise AssertionError("missing key did not raise")
    finally:
        os.environ.update(saved)
        if prior_env_file is None:
            os.environ.pop(ENV_FILE_ENV, None)
        else:
            os.environ[ENV_FILE_ENV] = prior_env_file
        blocker.unlink(missing_ok=True)
    print("ok  missing OPENCODE_API_KEY fails fast")


def check_list_models() -> None:
    server, base, calls = _start_server(
        lambda record: (
            200,
            {"object": "list", "data": [{"id": DEFAULT_MODEL, "name": "GLM 5.3"}]},
            {},
        )
    )
    try:
        client = _client(base)
        models = client.list_models()
        assert models == [{"id": DEFAULT_MODEL, "name": "GLM 5.3"}]
        record = calls[0]
        assert record["method"] == "GET"
        assert record["path"] == "/v1/models"
    finally:
        server.shutdown()
    print("ok  GET models catalog")


def check_async_chat() -> None:
    server, base, calls = _start_server(lambda record: (200, _chat_payload(), {}))
    try:
        client = _client(base)
        response = asyncio.run(client.achat("hello"))
        assert response.text == "hi from go"
    finally:
        server.shutdown()
    print("ok  achat works from an event loop")


def check_tool_call_shape() -> None:
    payload = _chat_payload()
    payload["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "pick_move",
                    "arguments": "{\"move\": \"u-turn\"}",
                },
            }
        ],
    }
    payload["choices"][0]["finish_reason"] = "tool_calls"
    server, base, calls = _start_server(lambda record: (200, payload, {}))
    try:
        response = _client(base).chat("hello")
        assert response.text == ""
        assert response.finish_reason == "tool_calls"
        assert response.tool_calls[0]["function"]["name"] == "pick_move"
    finally:
        server.shutdown()
    print("ok  tool_calls surfaced (empty text)")


CHECKS = [
    check_default_urls,
    check_chat_request_and_usage,
    check_string_prompt_and_message_list,
    check_retry_on_429,
    check_http_error_mapping,
    check_missing_api_key,
    check_list_models,
    check_async_chat,
    check_tool_call_shape,
]


def main() -> int:
    for check in CHECKS:
        check()
    print(f"\nall {len(CHECKS)} llm client checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
