"""Offline checks for the Pi-agent runtime (no server or OpenCode key)."""
from __future__ import annotations

import os

from pi_agent.config import SYSTEM_PROMPT, agent_config_from_env
from pi_agent.damage import damage_report
from pi_agent.player import _forced_order
from pi_agent.prompt import build_actions, build_prompt, build_repair_prompt, parse_answer
from pi_agent.provider import opencode_model, opencode_provider
from pi_agent.tools import TurnContext, build_tools

MODEL = "deepseek-v4.1-flash"


class FakePokemon:
    species = "greattusk"

    def __init__(self) -> None:
        self.types = ["Ground", "Fighting"]
        self.current_hp = 100
        self.max_hp = 100
        self.current_hp_fraction = 1.0
        self.status = None
        self.item = "boots"
        self.boosts = {}
        self.stats = {"atk": 120, "def": 80, "spa": 60, "spd": 60, "spe": 100}
        self.fainted = False


class FakeMove:
    id = "earthquake"
    name = "Earthquake"
    base_power = 100
    category = "physical"
    priority = 0

    def __init__(self) -> None:
        self.type = "ground"
        self.flags = "offensive"


class FakeBattle:
    turn = 1
    active_pokemon = FakePokemon()
    opponent_active_pokemon = FakePokemon()
    available_moves = [FakeMove()]
    available_switches = []
    team = {}
    opponent_team = {}
    weather = {}
    fields = {}
    side_conditions = {}
    opponent_side_conditions = {}
    can_terastallize = False


class FakeBattleWithChoice(FakeBattle):
    available_moves = [FakeMove(), FakeMove()]


class FakePlayer:
    def create_order(self, order, **kwargs):
        return ("order", order, kwargs)

    def choose_random_move(self, battle):
        return ("random",)


def check_parsing() -> None:
    assert parse_answer('```json\n{"move": "earthquake"}\n```') == {"move": "earthquake"}
    chained = 'I first considered {"reason": "chip damage"} then {"move": "stealthrock"}'
    assert parse_answer(chained) == {"move": "stealthrock"}
    assert parse_answer('{"action": "switch", "target": "greattusk"}') == {
        "action": "switch",
        "target": "greattusk",
    }
    assert parse_answer("no json here") is None
    assert parse_answer('{"reason": "only prose"}') is None


def check_forced_orders() -> None:
    player = FakePlayer()
    battle = FakeBattle()
    forced = _forced_order(player, battle)
    assert forced is not None and forced[0] == "order", forced
    assert _forced_order(player, FakeBattleWithChoice()) is None


def check_repair_prompt() -> None:
    actions = build_actions(FakeBattle())
    prompt = build_repair_prompt(actions, "I would use Earthquake")
    assert "Legal moves: earthquake" in prompt, prompt
    assert '{"move":"earthquake"}' in prompt, prompt
    assert "Previous answer: I would use Earthquake" in prompt, prompt


def check_damage_report() -> None:
    report = damage_report(
        move="earthquake",
        move_type="Ground",
        category="physical",
        base_power=100,
        accuracy=100,
        priority=0,
        target="Great Tusk",
        multiplier=2.0,
        stab=True,
        percent=142.0,
        outspeed=1.0,
    )
    assert "KO: yes" in report, report
    assert "type x2.0" in report, report
    assert "STAB yes" in report, report
    assert "you move first" in report, report

    immune = damage_report(
        move="earthquake",
        move_type="Ground",
        category="physical",
        base_power=100,
        accuracy=100,
        priority=0,
        target="Corviknight",
        multiplier=0.0,
        stab=False,
        percent=0.0,
        outspeed=0.0,
    )
    assert "immune" in immune and "the target moves first" in immune, immune

    # poke-env's fork reports accuracy as a fraction and MoveCategory as an
    # object whose str() is "SPECIAL (move category) object"; both must be
    # normalised before the text reaches the model.
    class FakeCategory:
        name = "SPECIAL"

    fractional = damage_report(
        move="dracometeor",
        move_type="Dragon",
        category=FakeCategory(),
        base_power=130,
        accuracy=0.9,
        priority=0,
        target="Moltres",
        multiplier=1.0,
        stab=True,
        percent=207.0,
        outspeed=1.0,
    )
    assert "special" in fractional and "(move category)" not in fractional, fractional
    assert "90% accuracy" in fractional, fractional

    always = damage_report(
        move="protect",
        move_type="Normal",
        category="status",
        base_power=0,
        accuracy=True,
        priority=4,
        target="Moltres",
        multiplier=1.0,
        stab=False,
        percent=0.0,
        outspeed=0.0,
    )
    assert "always hits" in always and "priority +4" in always, always


def check_fixed_damage_moves() -> None:
    from pi_agent.damage import estimate_damage

    class FakeMove:
        id = "ruination"
        name = "Ruination"
        base_power = 0
        type = None
        category = "special"

    text = estimate_damage(object(), FakeMove(), object(), None)
    assert "fixed or formula-based damage" in text, text
    assert "status move" not in text, text


def check_config_clamps() -> None:
    key = "PI_AGENT_MAX_TOKENS"
    previous = os.environ.get(key)
    os.environ[key] = "999999"
    try:
        config = agent_config_from_env(rag=False, model=MODEL, kb_root="")
        assert config.max_tokens == 32768, config.max_tokens
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def main() -> int:
    battle = FakeBattle()
    context = agent_config_from_env(rag=False, model=MODEL, kb_root="")
    actions = build_actions(battle)
    assert actions and actions[0].name == "earthquake", actions
    assert build_prompt(battle).startswith("TURN 1"), build_prompt(battle)
    turn_context = TurnContext.for_turn(
        battle=battle,
        player=None,
        knowledge=None if context.rag else None,
    )
    assert build_tools(turn_context, context)[0].declaration().name == "get_battle_state"
    provider = opencode_provider(context)
    model = opencode_model(provider, context)
    assert model.id == MODEL, model
    assert model.provider == "opencode-go", model.provider
    check_parsing()
    check_forced_orders()
    check_repair_prompt()
    check_damage_report()
    check_fixed_damage_moves()
    check_config_clamps()
    assert "no switch tool" in SYSTEM_PROMPT.lower(), SYSTEM_PROMPT
    print("all pi_agent offline checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
