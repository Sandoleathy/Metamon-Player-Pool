"""Damage, type, KO and turn-order evidence for the ``calculate_damage`` tool.

The numbers come from metamon's own baseline helpers
(``metamon.baselines.base.Baseline``), which implement the same simplified
Showdown damage equation the heuristic baselines use.  Binding those methods
keeps the tool consistent with the rest of the project instead of maintaining a
second damage formula.

``damage_report`` is a pure formatter so the offline check can exercise the
model-facing text without a live battle; ``estimate_damage`` adds the
poke-env/metamon lookups on top.
"""
from __future__ import annotations

import math
from typing import Any

# Baseline helpers this tool reuses; keep the list explicit so a rename in
# metamon fails loudly here instead of silently degrading the tool.
_HELPER_METHODS = (
    "type_advantage",
    "infer_stats",
    "current_accuracy",
    "_stat_from_base_stats_early_gens",
    "_stat_from_base_stats_gen3plus",
    "damage_equation",
    "crit_chance",
    "expected_damage",
    "outspeed_chance",
)


# Moves whose damage is fixed or derived from HP/level rather than base power.
# Their Gen 9 data entry has base_power 0, which would otherwise be reported as
# a status move even though they do deal damage.
FIXED_DAMAGE_MOVES = frozenset(
    {
        "ruination",
        "superfang",
        "seismictoss",
        "nightshade",
        "counter",
        "mirrorcoat",
        "finalgambit",
        "endeavor",
    }
)


class DamageMath:
    """Only the helper methods of ``metamon.baselines.base.Baseline``."""

    def __init__(self) -> None:
        from metamon.baselines.base import Baseline

        self.get_gen_format = Baseline.get_gen_format.__get__(self, type(self))
        for name in _HELPER_METHODS:
            setattr(self, name, getattr(Baseline, name).__get__(self, type(self)))


def max_hp_of(mon: Any) -> float:
    value = getattr(mon, "max_hp", None) or getattr(mon, "current_hp", None)
    return float(value) if value else 100.0


def _accuracy_text(accuracy: Any) -> str:
    if accuracy is True or accuracy is None:
        return "always hits"
    try:
        value = float(accuracy)
    except (TypeError, ValueError):
        return "accuracy unknown"
    if 0.0 < value <= 1.0:
        # poke-env's fork stores accuracy as a fraction (1.0 == 100%).
        value *= 100.0
    return f"{value:g}% accuracy"


def _category_text(category: Any) -> str:
    """``MoveCategory`` prints as "SPECIAL (move category) object"; use its name."""
    name = getattr(category, "name", None)
    text = str(name if name is not None else category)
    return text.replace("MoveCategory.", "").lower()


def _speed_text(outspeed: float) -> str:
    if outspeed >= 0.999:
        return "turn order: you move first"
    if outspeed <= 0.001:
        return "turn order: the target moves first"
    return f"turn order: uncertain, you move first about {outspeed:.0%} of the time"


def damage_report(
    *,
    move: str,
    move_type: str,
    category: str,
    base_power: int,
    accuracy: Any,
    priority: int,
    target: str,
    multiplier: float,
    stab: bool,
    percent: float,
    outspeed: float,
) -> str:
    """Format one damage estimate as a short, model-readable block."""
    priority_text = "normal priority" if not priority else f"priority {priority:+d}"
    percent = max(0.0, float(percent))
    lines = [
        f"{move} ({move_type}, {_category_text(category)}, {base_power} BP, "
        f"{_accuracy_text(accuracy)}, {priority_text}) against {target}",
        f"expected damage about {percent:.0f}% of max HP | "
        f"type x{multiplier:.1f} | STAB {'yes' if stab else 'no'}",
    ]
    if multiplier == 0:
        lines.append("KO: no, the target is immune to this move type")
    elif percent >= 100:
        lines.append("KO: yes, expected damage covers the target's current HP")
    elif percent > 0:
        lines.append(f"KO: no, about {math.ceil(100.0 / percent)} hits from current HP")
    else:
        lines.append("KO: no, expected damage is zero")
    lines.append(_speed_text(outspeed))
    return "\n".join(lines)


def estimate_damage(user: Any, move: Any, target: Any, battle: Any) -> str:
    """Damage / type / KO / speed evidence for one legal move, as text."""
    move_name = str(getattr(move, "id", None) or getattr(move, "name", "?"))
    base_power = int(getattr(move, "base_power", 0) or 0)
    move_type = str(getattr(getattr(move, "type", None), "name", "?"))
    if move_type.isupper():
        move_type = move_type.capitalize()
    category = getattr(move, "category", "?")

    if move_name.lower() in FIXED_DAMAGE_MOVES:
        return (
            f"{move_name} deals fixed or formula-based damage that the simplified "
            f"damage equation does not model (its base power is 0 in the data), so no "
            f"percentage is available; reason about it from its own rule instead."
        )

    if base_power <= 0:
        return (
            f"{move_name} is a status move (no direct damage); consider its boost, "
            f"status, hazard, healing or field effect instead."
        )

    target_name = str(getattr(target, "species", "the target") or "the target")
    try:
        math_ = DamageMath()
        multiplier = float(math_.type_advantage(move.type, target, battle))
        damage = float(math_.expected_damage(user, move, target, battle))
        percent = 100.0 * damage / max_hp_of(target)
        outspeed = float(math_.outspeed_chance(user, target, battle))
    except Exception as exc:  # noqa: BLE001 — the tool reports, never raises
        return f"damage unavailable ({type(exc).__name__}: {exc})"

    types = list(getattr(user, "types", None) or [])
    stab = getattr(move, "type", None) in types
    return damage_report(
        move=move_name,
        move_type=move_type,
        category=category,
        base_power=base_power,
        accuracy=getattr(move, "accuracy", None),
        priority=int(getattr(move, "priority", 0) or 0),
        target=target_name,
        multiplier=multiplier,
        stab=stab,
        percent=percent,
        outspeed=outspeed,
    )
