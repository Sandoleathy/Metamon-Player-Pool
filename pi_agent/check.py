"""Offline checks for the Pi-agent runtime (no server or OpenCode key)."""
from __future__ import annotations

from pi_agent.config import agent_config_from_env
from pi_agent.provider import opencode_model, opencode_provider
from pi_agent.prompt import build_actions, build_prompt
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
    print("all pi_agent offline checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
