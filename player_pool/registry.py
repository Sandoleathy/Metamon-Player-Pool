"""Player registry: usernames, checkpoints and human-readable metadata.

Everything the LLM agent needs lives in the generated ``players.json``.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

FORMAT = "gen9ou"
TEAM_SET = "competitive"
SERVER_URL = "ws://localhost:8000/showdown/websocket"
USERNAME_PREFIX = "MM-"
PLAYERS_JSON = Path(__file__).parent / "players.json"


@dataclass
class PlayerSpec:
    username: str
    display_name: str
    kind: str  # "rl" | "heuristic"
    description: str
    agent: Optional[str] = None  # pretrained model name (RL only)
    baseline: Optional[str] = None  # metamon baseline class name (heuristic only)
    checkpoint: Optional[int] = None
    params: Optional[str] = None
    strength: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def as_json(self, battle_format: str) -> dict:
        data = asdict(self)
        data["format"] = battle_format
        data["challenge"] = f"/challenge {self.username}, {battle_format}"
        return data


# --------------------------------------------------------------------------
# RL policies: pretrained metamon checkpoints that were trained for Gen 9 OU.
# Checkpoint epochs match ``metamon/rl/pretrained.py`` defaults.
# --------------------------------------------------------------------------
RL_PLAYERS: List[PlayerSpec] = [
    PlayerSpec(
        username="MM-Kakuna",
        display_name="Kakuna",
        kind="rl",
        agent="Kakuna",
        checkpoint=34,
        params="142M",
        description="Best public metamon policy (Dec 2025). Superkazam finetuned on 7.8M "
        "additional high-temperature self-play battles.",
        strength="Gen9OU GXE ~71% vs humans with sample teams (paper Table 1).",
        tags=["strongest", "gen9ou", "transformer"],
    ),
    PlayerSpec(
        username="MM-Kadabra3",
        display_name="Kadabra3",
        kind="rl",
        agent="Kadabra3",
        checkpoint=20,
        params="57M",
        description="PokéAgent Challenge organizer baseline; played as PAC-MM-Wildcard / "
        "PAC-MM-Mystery during qualification.",
        strength="Gen9OU GXE ~64% vs humans; would have been #3 seed in the Gen9OU qualifier.",
        tags=["strong", "gen9ou", "organizer-baseline"],
    ),
    PlayerSpec(
        username="MM-Kadabra4",
        display_name="Kadabra4",
        kind="rl",
        agent="Kadabra4",
        checkpoint=50,
        params="57M",
        description="Final Kadabra trained on the full 11.6M self-play + parsed-replays-v4 corpus.",
        strength="#2 Gen9OU seed behind FoulPlay if it had been public; stronger than Kadabra3 "
        "on diverse teams.",
        tags=["strong", "gen9ou", "organizer-baseline"],
    ),
    PlayerSpec(
        username="MM-Alakazam",
        display_name="Alakazam",
        kind="rl",
        agent="Alakazam",
        checkpoint=8,
        params="57M",
        description="Last model of the PokéAgent Challenge effort; Kadabra4 finetuned on a "
        "dataset patched for tera-type visibility.",
        strength="Slightly better than Kadabra3/4 on the current (patched) server, more "
        "robust across diverse teams.",
        tags=["strong", "gen9ou"],
    ),
    PlayerSpec(
        username="MM-Abra",
        display_name="Abra",
        kind="rl",
        agent="Abra",
        checkpoint=40,
        params="57M",
        description="Best publicly available Gen9OU agent during the NeurIPS 2025 PokéAgent "
        "Challenge; the basis of many third-party submissions.",
        strength="Gen9OU GXE ~50% vs humans with sample teams.",
        tags=["gen9ou", "public-baseline"],
    ),
    PlayerSpec(
        username="MM-Superkazam",
        display_name="Superkazam",
        kind="rl",
        agent="Superkazam",
        checkpoint=50,
        params="142M",
        description="Revisits the PokéAgent Challenge dataset at ~142M parameters "
        "(11.6M self-play + 4M human battles).",
        strength="Between the 57M Kadabra line and Kakuna.",
        tags=["gen9ou", "transformer"],
    ),
    PlayerSpec(
        username="MM-Kadabra2",
        display_name="Kadabra2",
        kind="rl",
        agent="Kadabra2",
        checkpoint=44,
        params="57M",
        description="Mid-challenge policy; played on the ladder as PAC-MM-Alakazam.",
        strength="Held the top organizer Gen9OU rank at the end of the summer 2025 practice ladder.",
        tags=["gen9ou", "organizer-baseline"],
    ),
    PlayerSpec(
        username="MM-Kadabra",
        display_name="Kadabra",
        kind="rl",
        agent="Kadabra",
        checkpoint=46,
        params="57M",
        description="Early PokéAgent Challenge self-play policy (gens 1-4 and 9).",
        strength="Held the top organizer Gen9OU rank for most of the summer 2025 practice ladder.",
        tags=["gen9ou", "organizer-baseline"],
    ),
    PlayerSpec(
        username="MM-Minikazam",
        display_name="Minikazam",
        kind="rl",
        agent="Minikazam",
        checkpoint=40,
        params="4.7M",
        description="Tiny RNN trained on parsed-replays-v4 plus ~5M self-play battles; a cheap "
        "starting point for finetuning.",
        strength="Weakest of the modern RL policies, but cheap to run on CPU.",
        tags=["gen9ou", "small", "rnn"],
    ),
    PlayerSpec(
        username="MM-SmallRLGen9",
        display_name="SmallRLGen9Beta",
        kind="rl",
        agent="SmallRLGen9Beta",
        checkpoint=24,
        params="~15M",
        description="First Gen9-compatible prototype; trained only on human replays "
        "(parsed-replays v3).",
        strength="Roughly LargeRL-level in gens 1-4, weakest Gen9OU policy in the pool.",
        tags=["gen9ou", "prototype", "small"],
    ),
]


# --------------------------------------------------------------------------
# Heuristic / IL baselines registered in metamon.baselines (16 total).
# --------------------------------------------------------------------------
HEURISTIC_PLAYERS: List[PlayerSpec] = [
    PlayerSpec(
        username="MM-Random",
        display_name="RandomBaseline",
        kind="heuristic",
        baseline="RandomBaseline",
        description="Uniformly picks a legal move (or switch). Measures the floor of the pool.",
        strength="Very weak; sanity-check opponent.",
        tags=["heuristic", "random"],
    ),
    PlayerSpec(
        username="MM-BugCatcher",
        display_name="BugCatcher",
        kind="heuristic",
        baseline="BugCatcher",
        description="Actively bad trainer: always picks the least damaging move and the worst "
        "type matchup when forced to switch.",
        strength="Very weak; sanity-check opponent.",
        tags=["heuristic", "random"],
    ),
    PlayerSpec(
        username="MM-MaxBP",
        display_name="MaxBPBaseline",
        kind="heuristic",
        baseline="MaxBPBaseline",
        description="Usually picks the available move with the highest base power (10% random).",
        strength="Weak; no type or stat awareness.",
        tags=["heuristic", "greedy"],
    ),
    PlayerSpec(
        username="MM-Gen1Trainer",
        display_name="Gen1Trainer",
        kind="heuristic",
        baseline="Gen1Trainer",
        description="Gen-1-flavoured trainer: random moves, first available switch when forced.",
        strength="Weak.",
        tags=["heuristic", "gen1"],
    ),
    PlayerSpec(
        username="MM-Gen1Switch",
        display_name="Gen1TrainerGoodSwitching",
        kind="heuristic",
        baseline="Gen1TrainerGoodSwitching",
        description="Gen1Trainer with type/speed-aware switching when forced to switch.",
        strength="Weak.",
        tags=["heuristic", "gen1"],
    ),
    PlayerSpec(
        username="MM-Gen1Boss",
        display_name="Gen1BossAI",
        kind="heuristic",
        baseline="Gen1BossAI",
        description="Emulates the Gen 1 'Good AI': random moves, but prefers stat boosts on the "
        "first turn and super-effective moves.",
        strength="Weak; Gen 1 logic applied to Gen 9.",
        tags=["heuristic", "gen1"],
    ),
    PlayerSpec(
        username="MM-BasicSwitch",
        display_name="BasicSwitcher",
        kind="heuristic",
        baseline="BasicSwitcher",
        description="Greedy damage dealer that switches on bad type matchups.",
        strength="Low.",
        tags=["heuristic", "damage"],
    ),
    PlayerSpec(
        username="MM-Grunt",
        display_name="Grunt",
        kind="heuristic",
        baseline="Grunt",
        description="Maximally offensive: picks the highest estimated damage move using the "
        "damage formula and type chart; best type matchup when switching.",
        strength="Moderate-low; no status/boosting play.",
        tags=["heuristic", "damage"],
    ),
    PlayerSpec(
        username="MM-GymLeader",
        display_name="GymLeader",
        kind="heuristic",
        baseline="GymLeader",
        description="Grunt plus health awareness: boosts when healthy, heals when low.",
        strength="Moderate-low; a step above Grunt.",
        tags=["heuristic", "damage", "healing"],
    ),
    PlayerSpec(
        username="MM-SmogonSwitch",
        display_name="SmogonSwitcher",
        kind="heuristic",
        baseline="SmogonSwitcher",
        description="Grunt with Smogon-style switching heuristics.",
        strength="Moderate-low.",
        tags=["heuristic", "damage", "switching"],
    ),
    PlayerSpec(
        username="MM-BaseRNN",
        display_name="BaseRNN",
        kind="heuristic",
        baseline="BaseRNN",
        description="Small RNN imitation-learning policy trained on an early parsed-replay "
        "dataset; CPU inference.",
        strength="Low; pre-Gen9 IL checkpoint.",
        tags=["il", "rnn"],
    ),
    PlayerSpec(
        username="MM-WinsOnlyRNN",
        display_name="WinsOnlyRNN",
        kind="heuristic",
        baseline="WinsOnlyRNN",
        description="BaseRNN variant trained only on human wins.",
        strength="Low; pre-Gen9 IL checkpoint.",
        tags=["il", "rnn"],
    ),
    PlayerSpec(
        username="MM-MiniRNN",
        display_name="MiniRNN",
        kind="heuristic",
        baseline="MiniRNN",
        description="Smaller RNN imitation-learning baseline.",
        strength="Low; pre-Gen9 IL checkpoint.",
        tags=["il", "rnn"],
    ),
    PlayerSpec(
        username="MM-KaizoPlus",
        display_name="KaizoPlus",
        kind="heuristic",
        baseline="KaizoPlus",
        description="Extended version of the Emerald Kaizo ROM-hack AI with additional rules.",
        strength="Moderate; large handwritten ruleset.",
        tags=["heuristic", "kaizo"],
    ),
    PlayerSpec(
        username="MM-EmeraldKaizo",
        display_name="EmeraldKaizo",
        kind="heuristic",
        baseline="EmeraldKaizo",
        description="Port of the AI from the Pokémon Emerald Kaizo ROM hack: scores every "
        "option with a large handwritten ruleset.",
        strength="Moderate; the strongest heuristic baseline in the paper's composite.",
        tags=["heuristic", "kaizo"],
    ),
    PlayerSpec(
        username="MM-PokeEnv",
        display_name="PokeEnvHeuristic",
        kind="heuristic",
        baseline="PokeEnvHeuristic",
        description="The SimpleHeuristicsPlayer shipped with poke-env (type/damage heuristics).",
        strength="Moderate-low; the standard poke-env baseline.",
        tags=["heuristic", "poke-env"],
    ),
]


def all_specs() -> List[PlayerSpec]:
    return RL_PLAYERS + HEURISTIC_PLAYERS


def specs_by_username(prefix: str = USERNAME_PREFIX) -> List[PlayerSpec]:
    """Return specs whose username starts with ``prefix`` (identity by default)."""
    return [s for s in all_specs() if s.username.startswith(prefix)]


def write_players_json(
    path: Path | str = PLAYERS_JSON,
    battle_format: str = FORMAT,
    specs: Optional[List[PlayerSpec]] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Write the registry consumed by the LLM agent."""
    path = Path(path)
    specs = specs if specs is not None else all_specs()
    payload = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "server": {
            "websocket_url": SERVER_URL,
            "format": battle_format,
            "team_set": TEAM_SET,
        },
        "usage": (
            "Pick a username and challenge it in the given format, e.g. "
            "'/challenge MM-Kakuna, gen9ou'. Every player accepts challenges from "
            "anyone and plays until it loses/forfeits or the battle ends. RL "
            "players load their checkpoint on the first challenge they receive."
        ),
        "players": [spec.as_json(battle_format) for spec in specs],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    out = write_players_json()
    print(f"wrote {out}")
