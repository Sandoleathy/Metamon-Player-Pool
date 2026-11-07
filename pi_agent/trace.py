"""Local decision trace (JSONL) for every turn of an agent battle.

One line per decided turn.  Besides the prompt and the parsed answer the record
keeps the tool activity, because "what did the agent look up before choosing"
is the evidence a RAG/no-RAG comparison needs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_ANSWER = 4000
_MAX_TOOL_ARGUMENTS = 400
_MAX_TOOL_RESULT = 1200


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[+{len(text) - limit} chars]"


def _content_text(message: Any) -> str:
    parts = [
        getattr(block, "text", "")
        for block in (getattr(message, "content", None) or [])
        if getattr(block, "text", "")
    ]
    return "\n".join(parts)


def _result_details(result: Any) -> dict[str, Any]:
    """Pull usage, thinking, stop reason and tool activity out of a run result.

    Duck-typed on purpose: the trace must keep working if pi-python renames its
    message classes, and it must not import pi-python at module import time.
    """
    details: dict[str, Any] = {
        "usage": {},
        "thinking": None,
        "stop_reason": None,
        "tools": [],
    }
    messages = list(getattr(result, "messages", None) or [])
    details["usage"] = dict(getattr(result, "usage", None) or {})

    results_by_call: dict[str, str] = {}
    for message in messages:
        role = getattr(message, "role", None)
        if type(message).__name__ == "ToolResultMessage" or role == "tool_result":
            call_id = getattr(message, "call_id", "") or ""
            results_by_call[call_id] = _clip(_content_text(message), _MAX_TOOL_RESULT)

    thinking: list[str] = []
    for message in messages:
        if type(message).__name__ != "AssistantMessage":
            continue
        usage = getattr(message, "usage", None)
        if usage:
            details["usage"] = dict(usage)
        details["stop_reason"] = getattr(message, "stop_reason", None) or details["stop_reason"]
        for block in getattr(message, "content", None) or []:
            kind = type(block).__name__
            if kind == "ThinkingContent":
                text = (getattr(block, "thinking", "") or "").strip()
                if text:
                    thinking.append(text)
            elif kind == "ToolCall":
                details["tools"].append(
                    {
                        "name": getattr(block, "name", "?"),
                        "arguments": _clip(getattr(block, "arguments", ""), _MAX_TOOL_ARGUMENTS),
                        "result": results_by_call.get(getattr(block, "id", "") or "", ""),
                    }
                )

    if thinking:
        details["thinking"] = "\n\n".join(thinking)
    return details


@dataclass
class TraceWriter:
    slug: str
    path: Path

    @classmethod
    def for_player(cls, root: Path, username: str) -> TraceWriter:
        root.mkdir(parents=True, exist_ok=True)
        slug = f"{time.strftime('%Y%m%d-%H%M%S')}-{username}"
        return cls(slug=slug, path=root / f"{slug}.jsonl")

    def _append(self, record: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
        except OSError as exc:
            # A read-only trace directory must not end a battle.
            print(f"could not write pi_agent trace {self.path}: {exc}", flush=True)

    def _where(self, battle: Any) -> dict[str, Any]:
        return {
            "battle": getattr(battle, "battle_tag", None),
            "turn": getattr(battle, "turn", None),
            "opponent": getattr(battle, "opponent_username", None),
        }

    def record(
        self,
        *,
        battle: Any,
        prompt: str,
        actions: list[str],
        result: Any,
        answer: str,
        parsed: Any,
        order: str,
        rag: bool,
        attempts: list[dict[str, Any]] | None = None,
        latency: float | None = None,
    ) -> None:
        details = _result_details(result)
        record = {
            "slug": self.slug,
            "ts": time.time(),
            **self._where(battle),
            "rag": rag,
            "prompt_chars": len(prompt),
            "actions": actions,
            "answer": _clip(answer, _MAX_ANSWER),
            "parsed": parsed,
            "chosen": order,
            "chosen_kind": "model" if attempts and any(a.get("chosen") for a in attempts) else "fallback",
            "attempts": attempts or [],
            "usage": details["usage"],
            "thinking": details["thinking"],
            "tools": details["tools"],
            "stop_reason": details["stop_reason"],
            "latency_s": round(latency, 3) if latency is not None else None,
        }
        self._append(record)
        print(f"[{self.slug}] TURN {record['turn']} -> {order}", flush=True)

    def record_forced(self, battle: Any, order: Any, reason: str = "forced") -> None:
        """A turn decided without a model call (single legal action)."""
        message = (getattr(order, "message", "") or str(order)) if order is not None else "none"
        record = {
            "slug": self.slug,
            "ts": time.time(),
            **self._where(battle),
            "forced": True,
            "forced_reason": reason,
            "chosen": message,
            "chosen_kind": "forced",
        }
        self._append(record)
        print(f"[{self.slug}] TURN {record['turn']} -> {message} (forced)", flush=True)
