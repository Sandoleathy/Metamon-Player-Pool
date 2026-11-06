"""Chat client for the opponent agent, with OpenCode Go as the default provider.

The RAG agent that challenges the pool needs an LLM endpoint. OpenCode Console
serves one OpenAI-compatible surface for every Go text model:

    POST https://opencode.ai/inference/go/openai/v1/chat/completions
    Authorization: Bearer oc_sk_...
    x-opencode-session: ses_...

The session header is mandatory for Go (a request without it fails with
``400 MissingSessionID``), and OpenCode asks non-OpenCode clients to identify
themselves with a real User-Agent instead of a generic HTTP-library name. Both
are handled here, so callers only deal with messages and text.

Only the standard library is used; this module adds no dependency.

Environment:

    OPENCODE_API_KEY       personal/service-account key (``OPENCODE_GO_API_KEY``
                           is accepted as a fallback)
    OPENCODE_MODEL         default model, e.g. ``glm-5.3``
    OPENCODE_BASE_URL      override the provider base URL
    OPENCODE_SESSION_ID    stable session id; generated per client when unset

CLI::

    python -m player_pool.llm_client --model glm-5.3 --prompt "Say hi"
    python -m player_pool.llm_client --list-models
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

CLIENT_VERSION = "0.1.0"
USER_AGENT = f"metamon-player-pool/{CLIENT_VERSION}"

DEFAULT_PROVIDER = "opencode-go"
DEFAULT_MODEL = "glm-5.3"

#: Provider roots. Paths below are appended by the client.
PROVIDER_BASE_URLS = {
    # Go / Go Plus / Go Teams
    "opencode-go": "https://opencode.ai/inference/go",
    # pay-as-you-go balance
    "opencode": "https://opencode.ai/inference",
}

API_KEY_ENVS = ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY")
MODEL_ENV = "OPENCODE_MODEL"
BASE_URL_ENV = "OPENCODE_BASE_URL"
SESSION_ENV = "OPENCODE_SESSION_ID"

CHAT_PATH = "/openai/v1/chat/completions"
MODELS_PATH = "/v1/models"

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

Messages = Union[str, Sequence[Mapping[str, Any]]]


class LLMError(RuntimeError):
    """A failed inference request."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        error_type: Optional[str] = None,
        body: Optional[bytes] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in RETRYABLE_STATUS


class MissingAPIKeyError(LLMError):
    """No OpenCode credential was configured."""


@dataclass
class Usage:
    """Token/usage numbers from one response.

    ``cost`` is what OpenCode charged (``0`` for Go usage inside the plan) and
    ``allowance`` is how many dollars of the Go monthly limit the request used.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    allowance: Optional[float] = None
    billing_source: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    session_id: str
    finish_reason: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "session_id": self.session_id,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost": self.usage.cost,
                "allowance": self.usage.allowance,
                "billing_source": self.usage.billing_source,
            },
        }


ENV_FILE_ENV = "PLAYER_POOL_ENV_FILE"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """A tiny ``.env`` loader: fills gaps, existing environment wins."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_project_env() -> None:
    override = os.environ.get(ENV_FILE_ENV)
    if override:
        _load_env_file(Path(override))
        return
    _load_env_file(REPO_ROOT / ".env")


def _first_env(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def new_session_id() -> str:
    """A fresh OpenCode session id (``ses_`` + 32 hex chars)."""
    return "ses_" + uuid.uuid4().hex


def _normalize_messages(
    messages: Messages, system: Optional[str]
) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        normalized: List[Dict[str, Any]] = [{"role": "user", "content": messages}]
    else:
        normalized = [dict(message) for message in messages]
    for message in normalized:
        if "role" not in message or "content" not in message:
            raise ValueError("every message needs 'role' and 'content'")
    if system:
        if normalized and normalized[0].get("role") == "system":
            existing = normalized[0].get("content") or ""
            normalized[0]["content"] = f"{system}\n\n{existing}".rstrip()
        else:
            normalized.insert(0, {"role": "system", "content": system})
    return normalized


def _extract_error(body: bytes) -> tuple[Optional[str], str]:
    text = body.decode("utf-8", "replace")
    try:
        data = json.loads(text)
    except ValueError:
        return None, text[:500]
    if not isinstance(data, dict):
        return None, text[:500]
    error = data.get("error", data)
    if isinstance(error, dict):
        message = error.get("message") or json.dumps(error)
        return error.get("type"), str(message)
    if isinstance(error, str):
        return None, error
    return None, text[:500]


def _retry_delay(retry_after: Optional[str], attempt: int) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            pass
    return min(2.0**attempt, 15.0) + random.uniform(0.0, 0.25)


class LLMClient:
    """Synchronous client; ``achat`` runs it on a worker thread.

    One client corresponds to one conversation session. Reuse it for the whole
    battle/agent run so OpenCode can route and cache the prompt consistently.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: str = DEFAULT_PROVIDER,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        user_agent: str = USER_AGENT,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        _load_project_env()
        if provider not in PROVIDER_BASE_URLS and not base_url:
            raise ValueError(
                f"unknown provider {provider!r}; use one of "
                f"{sorted(PROVIDER_BASE_URLS)} or pass base_url"
            )
        selected_url = base_url or os.environ.get(BASE_URL_ENV)
        self.provider = provider
        self.base_url = (selected_url or PROVIDER_BASE_URLS[provider]).rstrip("/")
        self.api_key = api_key if api_key is not None else _first_env(API_KEY_ENVS)
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.session_id = (
            session_id or os.environ.get(SESSION_ENV) or new_session_id()
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.extra_headers = dict(extra_headers or {})

    @property
    def chat_url(self) -> str:
        return self.base_url + CHAT_PATH

    @property
    def models_url(self) -> str:
        return self.base_url + MODELS_PATH

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.provider!r}, model={self.model!r}, "
            f"session_id={self.session_id!r})"
        )

    # -- HTTP ---------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "x-opencode-session": self.session_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError(
                "no OpenCode API key: set OPENCODE_API_KEY (or "
                "OPENCODE_GO_API_KEY, or add it to the repo's .env) — get a key "
                "at https://opencode.ai/console"
            )

    def _request_json(
        self, method: str, url: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        attempt = 0
        while True:
            request = urllib.request.Request(
                url,
                data=body,
                headers=self._headers(),
                method=method,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                error_body = exc.read()
                error_type, message = _extract_error(error_body)
                error = LLMError(
                    f"HTTP {exc.code}: {message or exc.reason}",
                    status=exc.code,
                    error_type=error_type,
                    body=error_body,
                )
                if attempt < self.max_retries and exc.code in RETRYABLE_STATUS:
                    time.sleep(_retry_delay(exc.headers.get("Retry-After"), attempt))
                    attempt += 1
                    continue
                raise error from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                if attempt < self.max_retries:
                    time.sleep(_retry_delay(None, attempt))
                    attempt += 1
                    continue
                raise
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise LLMError(
                    f"invalid JSON response (HTTP {status}): {exc}",
                    status=status,
                    body=raw,
                ) from None

    # -- API ----------------------------------------------------------------

    def chat(
        self,
        messages: Messages,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        """One chat-completions call.

        ``extra`` is merged into the JSON body, so ``tools``,
        ``response_format``, ``top_p`` and provider-specific fields pass
        through unchanged. Streaming is not implemented; ``stream`` is
        rejected instead of returning an unparsable body.
        """
        self._require_api_key()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages(messages, system),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            for reserved in ("model", "messages", "stream"):
                if reserved in extra:
                    raise ValueError(f"pass {reserved!r} as its own argument")
            payload.update(extra)

        data = self._request_json("POST", self.chat_url, payload)
        return self._parse_response(data)

    async def achat(self, messages: Messages, **kwargs: Any) -> LLMResponse:
        """Async wrapper; safe to call from the agent's event loop."""
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    def list_models(self) -> List[Dict[str, Any]]:
        """Model catalog for the configured provider."""
        data = self._request_json("GET", self.models_url)
        models = data.get("data")
        return models if isinstance(models, list) else []

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(
                "response contains no choices", body=json.dumps(data).encode()
            )

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping)
            )
        else:
            text = content or ""

        usage_raw = data.get("usage") or {}
        opencode = usage_raw.get("opencode") or {}
        usage = Usage(
            prompt_tokens=int(
                usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0
            ),
            completion_tokens=int(
                usage_raw.get(
                    "completion_tokens", usage_raw.get("output_tokens", 0)
                )
                or 0
            ),
            total_tokens=int(usage_raw.get("total_tokens", 0) or 0),
            cost=usage_raw.get("cost"),
            allowance=opencode.get("allowance"),
            billing_source=opencode.get("billing_source"),
            raw=usage_raw,
        )
        return LLMResponse(
            text=text,
            model=data.get("model") or self.model,
            session_id=self.session_id,
            finish_reason=choice.get("finish_reason"),
            tool_calls=list(message.get("tool_calls") or []),
            usage=usage,
            raw=data,
        )


def _read_prompt(argument: Optional[str]) -> str:
    if argument is not None:
        return argument
    if sys.stdin.isatty():
        raise SystemExit("no prompt: pass --prompt or pipe text on stdin")
    return sys.stdin.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Call OpenCode Go (or pay-as-you-go) chat completions."
    )
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER, choices=sorted(PROVIDER_BASE_URLS)
    )
    parser.add_argument("--model", default=None, help=f"default {DEFAULT_MODEL}")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the full reply")
    args = parser.parse_args(argv)

    try:
        client = LLMClient(
            args.model,
            provider=args.provider,
            base_url=args.base_url,
            session_id=args.session_id,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        if args.list_models:
            for model in client.list_models():
                print(f"{model.get('id', '?')}\t{model.get('name', '')}".rstrip())
            return 0
        response = client.chat(
            _read_prompt(args.prompt),
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
