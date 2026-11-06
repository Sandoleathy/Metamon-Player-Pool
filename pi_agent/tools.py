"""Pi tools for one battle turn.

Every tool is a Python function wrapped by ``pi-python-core``; the schema comes
from the signature and the docstring.  Each tool returns a plain string so the
model can read it, and every failure is reported as data instead of crashing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig
from .prompt import ActionChoice, build_actions, build_prompt, format_actions

_CLEAN_ID = re.compile(r"[^A-Za-z0-9]+")


@dataclass
class TurnContext:
    battle: Any
    player: Any
    actions: list[ActionChoice] = field(default_factory=list)
    knowledge: Any | None = None

    @classmethod
    def for_turn(cls, battle: Any, player: Any, knowledge: Any | None) -> TurnContext:
        return cls(
            battle=battle,
            player=player,
            actions=build_actions(battle),
            knowledge=knowledge,
        )

    def prompt(self) -> str:
        return build_prompt(self.battle)

    def find_move(self, move_id: str) -> Any:
        wanted = _CLEAN_ID.sub("_", (move_id or "").lower()).strip("_")
        for action in self.actions:
            if action.kind == "move" and action.name == wanted:
                return action
        return None


def build_tools(context: TurnContext, config: AgentConfig) -> list[Any]:
    """Fresh, context-closed tools for one turn."""
    from pi_python import tool

    battle = context.battle

    @tool(name="get_battle_state")
    def get_battle_state() -> str:
        """Return the reduced battle-state text for this turn."""
        return context.prompt()

    @tool(name="get_available_actions")
    def get_available_actions() -> str:
        """Return the legal action list, including Terastallization legality."""
        return format_actions(context.actions)

    @tool(name="calculate_damage")
    def calculate_damage(move: str) -> str:
        """Estimate the user's active Pokemon's damage against the active opponent."""
        from .damage import estimate_damage_percent

        found = context.find_move(move)
        if found is None or found.base_power == 0:
            return f"move {move!r} is not a damaging legal move"
        user = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if user is None or opponent is None:
            return "no active Pokemon; cannot compute damage"
        return estimate_damage_percent(user, found.order, opponent, battle)

    tools: list[Any] = [get_battle_state, get_available_actions, calculate_damage]
    if config.rag and context.knowledge is not None:
        knowledge = context.knowledge

        @tool(name="search_knowledge")
        def search_knowledge(queries: list[str], category: str = "any", top_k: int = 1) -> str:
            """Search the local knowledge base for move, ability, item or rule cards."""
            from pi_agent.errors import KnowledgeError, format_knowledge_failure

            try:
                hits = knowledge.search(
                    queries=queries,
                    category=category,
                    top_k=top_k,
                )
            except KnowledgeError as exc:
                return format_knowledge_failure(exc)
            return knowledge.format_cards(hits)

        tools.append(search_knowledge)
    return tools


def tool_lines(context: TurnContext, config: AgentConfig) -> list[dict[str, Any]]:
    names = ["get_battle_state", "get_available_actions", "calculate_damage"]
    if config.rag and context.knowledge is not None:
        names.append("search_knowledge")
    return [
        {"name": name, "calls": 1, "budget": config.max_tool_calls} for name in names
    ]
