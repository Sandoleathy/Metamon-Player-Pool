"""Local decision trace (JSONL) for every turn of an agent battle."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TraceWriter:
    slug: str
    path: Path

    @classmethod
    def for_player(cls, root: Path, username: str) -> TraceWriter:
        root.mkdir(parents=True, exist_ok=True)
        slug = f"{time.strftime('%Y%m%d-%H%M%S')}-{username}"
        return cls(slug=slug, path=root / f"{slug}.jsonl")

    def record(
        self,
        battle: Any,
        prompt: str,
        actions: list[str],
        result: Any,
        answer: str,
        parsed: Any,
        order: str,
        rag: bool,
    ) -> None:
        usage: dict[str, Any] = {}
        thinking = ""
        for message in reversed(list(getattr(result, "messages", []) or [])):
            if type(message).__name__ != "AssistantMessage":
                continue
            usage = dict(getattr(message, "usage", None) or {})
            thinking = "\n".join(
                getattr(block, "thinking", "")
                for block in getattr(message, "content", [])
                if getattr(block, "type", None) == "thinking"
            )
            break
        record = {
            "slug": self.slug,
            "ts": time.time(),
            "battle": getattr(battle, "battle_tag", None),
            "turn": getattr(battle, "turn", None),
            "opponent": getattr(battle, "opponent_username", None),
            "rag": rag,
            "prompt_chars": len(prompt),
            "actions": actions,
            "answer": answer,
            "parsed": parsed,
            "chosen": order,
            "usage": usage,
            "thinking": thinking.strip() or None,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
        print(
            f"[{self.slug}] TURN {getattr(battle, 'turn', '-')} -> {order}",
            flush=True,
        )
