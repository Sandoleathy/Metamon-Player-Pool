"""State prompt, action list and answer parsing for one battle turn.

The prompt keeps only what a human sees plus simulator-computed numbers.  It
does not paraphrase move, ability or item effects; those are what the optional
knowledge tool is for.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

ACTION_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ActionChoice:
    kind: str  # "move" or "switch"
    name: str
    order: Any
    label: str = ""
    tera: bool = False
    base_power: int | None = None
    category: str = ""
    move_type: str = ""
    priority: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.name, "kind": self.kind, "label": self.label}
        if self.kind == "move":
            payload.update(
                {
                    "terastallized": self.tera,
                    "basePower": self.base_power,
                    "category": self.category,
                    "type": self.move_type,
                    "priority": self.priority,
                }
            )
        return payload


def _clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")


def display_for(mon: Any) -> str:
    for attr in ("species", "base_species", "name"):
        value = getattr(mon, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(mon)


def type_names(mon: Any) -> list[str]:
    types = getattr(mon, "types", None) or []
    return [getattr(t, "name", str(t)) for t in types if t is not None]


def boosts_for(mon: Any) -> dict[str, int]:
    try:
        return {k: int(v) for k, v in getattr(mon, "boosts", {}).items() if v}
    except Exception:
        return {}


def format_self(mon: Any) -> str:
    hp = getattr(mon, "current_hp", None)
    max_hp = getattr(mon, "max_hp", None) or 1
    try:
        hp_pct = round(100 * float(hp) / float(max_hp))
    except (TypeError, ValueError):
        hp_pct = -1
    item = getattr(mon, "item", None)
    return (
        f"{display_for(mon)} | types {'/'.join(type_names(mon)) or '?'} "
        f"| HP {hp_pct}% ({hp}/{max_hp}) | status {getattr(mon, 'status', None)} "
        f"| item {item or 'none'} | boosts {boosts_for(mon) or 'none'} "
        f"| stats {getattr(mon, 'stats', None) or 'unknown'}"
    )


def format_opponent(mon: Any) -> str:
    hp_pct = int(round(100 * getattr(mon, "current_hp_fraction", 0.0)))
    return (
        f"{display_for(mon)} | types {'/'.join(type_names(mon)) or '?'} "
        f"| HP {hp_pct}% | status {getattr(mon, 'status', None)} "
        f"| stats {getattr(mon, 'stats', None) or 'base:unknown'}"
    )


def build_prompt(battle: Any) -> str:
    """The text that goes into the user message; both RAG arms share it."""
    lines: list[str] = []
    our = getattr(battle, "active_pokemon", None)
    opp = getattr(battle, "opponent_active_pokemon", None)
    lines.append(f"TURN {getattr(battle, 'turn', '-')}")
    if our is not None:
        lines.append(f"OURS: {format_self(our)}")
    if opp is not None:
        lines.append(f"OPPONENT ACTIVE: {format_opponent(opp)}")

    our_team = [
        display_for(p)
        for p in (getattr(battle, "team", None) or {}).values()
        if getattr(p, "fainted", False) is False
    ]
    opp_revealed = [
        display_for(p) for p in (getattr(battle, "opponent_team", None) or {}).values()
    ]
    lines.append(f"OUR TEAM ALIVE: {our_team or []}")
    lines.append(f"OPPONENT REVEALED: {opp_revealed or ['unknown']}")

    weather = list(getattr(battle, "weather", None) or {})
    fields = list(getattr(battle, "fields", None) or {})
    side = getattr(battle, "side_conditions", None) or {}
    opp_side = getattr(battle, "opponent_side_conditions", None) or {}
    lines.append(
        f"FIELD: weather {[getattr(w, 'name', str(w)) for w in weather]} "
        f"| fields {[getattr(f, 'name', str(f)) for f in fields]}"
    )
    lines.append(
        f"HAZARDS: ours {[getattr(s, 'name', str(s)) for s in side if side.get(s, 0) > 0]} "
        f"| opponents {[getattr(s, 'name', str(s)) for s in opp_side if opp_side.get(s, 0) > 0]}"
    )
    lines.append(f"TERA LEGAL NOW: {getattr(battle, 'can_terastallize', False)}")
    return "\n".join(lines)


def build_actions(battle: Any) -> list[ActionChoice]:
    """Map poke-env objects into stable, name-based action choices."""
    actions: list[ActionChoice] = []
    for move in list(getattr(battle, "available_moves", None) or []) or []:
        move_id = _clean_id(getattr(move, "id", None) or getattr(move, "name", ""))
        actions.append(
            ActionChoice(
                kind="move",
                name=move_id,
                order=move,
                label=f"move {move_id}",
                tera=bool(getattr(battle, "can_terastallize", False)),
                base_power=getattr(move, "base_power", None),
                category=str(getattr(move, "category", "")).replace("MoveCategory.", ""),
                move_type=getattr(getattr(move, "type", None), "name", ""),
                priority=getattr(move, "priority", None),
            )
        )
    for mon in list(getattr(battle, "available_switches", None) or []) or []:
        species = _clean_id(display_for(mon))
        actions.append(
            ActionChoice(kind="switch", name=species, order=mon, label=f"switch {species}")
        )
    return actions


def format_actions(actions: Iterable[ActionChoice]) -> str:
    return json.dumps([a.as_dict() for a in actions], indent=2, sort_keys=True)


def highest_power_move(actions: Iterable[ActionChoice]) -> ActionChoice | None:
    moves = [a for a in actions if a.kind == "move"]
    if not moves:
        return None
    return max(moves, key=lambda a: a.base_power or 0)


def parse_answer(text: str | None) -> dict[str, Any] | None:
    """Tolerant extractor: the model may wrap JSON in prose or a code fence."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.count("{") > 1 or "\n" in candidate:
        match = ACTION_RE.search(candidate)
        if not match:
            return None
        candidate = match.group(0)
    try:
        data = json.loads(candidate)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
