"""Damage estimates reusing metamon's simplified Showdown damage equation.

This does not reimplement the formula.  It binds the helper methods from
``metamon.baselines.base.Baseline`` onto a tiny object, so the tool can run
without a connected poke-env player while keeping the same damage constants.
"""
from __future__ import annotations

from typing import Any


class DamageMath:
    """Only the helper methods of ``metamon.baselines.base.Baseline``."""

    def __init__(self) -> None:
        from metamon.baselines.base import Baseline

        self.get_gen_format = Baseline.get_gen_format.__get__(self, type(self))
        for name in (
            "type_advantage",
            "infer_stats",
            "current_accuracy",
            "_stat_from_base_stats_early_gens",
            "_stat_from_base_stats_gen3plus",
            "damage_equation",
            "crit_chance",
            "expected_damage",
        ):
            setattr(self, name, getattr(Baseline, name).__get__(self, type(self)))


def max_hp_of(mon: Any) -> float:
    value = getattr(mon, "max_hp", None) or getattr(mon, "current_hp", None)
    return float(value) if value else 100.0


def estimate_damage_percent(user: Any, move: Any, target: Any, battle: Any) -> str:
    """Expected damage against the active target as a percent of max HP."""
    try:
        math_ = DamageMath()
        damage = math_.expected_damage(user, move, target, battle)
        percent = 100 * float(damage) / max_hp_of(target)
        return f"expected damage ≈ {damage:.1f} HP ({max(0.0, round(percent))}% of max HP)"
    except Exception as exc:  # noqa: BLE001 — the tool must report, not crash
        return f"damage unavailable ({type(exc).__name__})"
