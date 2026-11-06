"""Agent configuration shared by the player and the CLI.

Defaults come from OpenCode Go (`opencode.ai/inference/go`); every value can be
overridden from the environment so the same code works with any
OpenAI-compatible chat endpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
SYSTEM_PROMPT = """You are a competitive Gen 9 OverUsed Pokemon player on a local Showdown server.

Each turn you receive the current battle state and must return exactly one JSON
object with the action for the turn:
- {"move": "<move-id>"} for a move
- {"switch": "<species-id>"} for a switch
- {"move": "<move-id>", "tera": true} only when Terastallization is legal
Add "reason": "<short justification>" when the decision is not forced.

Rules:
- Use the tools for state details, damage estimates and knowledge lookups before
  committing to a risky move.
- Return only the JSON object. No prose, code fences or extra keys.
"""


@dataclass(frozen=True)
class AgentConfig:
    base_url: str
    model: str
    api_key: str | None
    thinking: str
    rag: bool
    kb_name: str
    kb_root: str
    trace_dir: Path

    max_model_requests: int
    max_tool_calls: int
    run_timeout: float
    context_window: int
    max_tokens: int
    reasoning: bool


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def agent_config_from_env(
    *,
    rag: bool | None = None,
    model: str | None = None,
    trace_dir: Path | None = None,
    kb_root: str | None = None,
) -> AgentConfig:
    """Build the config for one session (one game), not one turn."""
    trace_root = trace_dir or Path(os.environ.get("PI_AGENT_TRACE_DIR", "pi_agent/logs"))
    return AgentConfig(
        base_url=os.environ.get("OPENCODE_BASE_URL") or DEFAULT_BASE_URL,
        model=model or _first_env("PI_AGENT_MODEL", "OPENCODE_MODEL") or DEFAULT_MODEL,
        api_key=_first_env("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY"),
        thinking=os.environ.get("PI_AGENT_THINKING", "off").strip().lower(),
        rag=_env_bool("PI_AGENT_RAG", True) if rag is None else rag,
        kb_name=os.environ.get("PI_KNOWLEDGE_NAME", "gen9ou"),
        kb_root=kb_root if kb_root is not None else os.environ.get("PI_KNOWLEDGE_ROOT", ""),
        trace_dir=trace_root,
        max_model_requests=int(os.environ.get("PI_AGENT_MAX_MODEL_REQUESTS", "8")),
        max_tool_calls=int(os.environ.get("PI_AGENT_MAX_TOOL_CALLS", "8")),
        run_timeout=float(os.environ.get("PI_AGENT_RUN_TIMEOUT", "240")),
        context_window=int(os.environ.get("PI_AGENT_CONTEXT_WINDOW", "128000")),
        max_tokens=int(os.environ.get("PI_AGENT_MAX_TOKENS", "8192")),
        reasoning=_env_bool("PI_AGENT_REASONING", True),
    )
