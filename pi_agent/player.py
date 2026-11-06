"""A tool-using LLM player backed by ``pi-python-core`` and the local pool."""
from __future__ import annotations

import logging
import os
from typing import Any

from .config import AgentConfig, SYSTEM_PROMPT, agent_config_from_env
from .knowledge import KnowledgeClient
from .prompt import (
    ActionChoice,
    build_actions,
    build_prompt,
    format_actions,
    highest_power_move,
    parse_answer,
)
from .provider import build_agent
from .tools import TurnContext, build_tools
from .trace import TraceWriter

def create_player(
    username: str,
    *,
    config: AgentConfig | None = None,
    knowledge: KnowledgeClient | None = None,
    trace: TraceWriter | None = None,
    battle_format: str = "gen9ou",
    team_set_name: str = "competitive",
    server_configuration: Any | None = None,
) -> Any:
    """Build a poke-env player whose ``choose_move`` runs the Pi agent loop.

    The LLM call happens synchronously inside poke-env's battle loop, so the
    connection is closed until the decision returns.  The timer is disabled to
    keep the battle stable while the model calls tools.
    """
    from poke_env import AccountConfiguration
    from poke_env.player import Player
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from player_pool.teams import get_local_teams

    if config is None:
        config = agent_config_from_env()
    if os.environ.get("PI_KNOWLEDGE_ROOT") != config.kb_root:
        os.environ["PI_KNOWLEDGE_ROOT"] = config.kb_root
    knowledge = (
        knowledge
        if knowledge is not None
        else (
            KnowledgeClient(name=config.kb_name, knowledge_root=config.kb_root)
            if config.rag
            else None
        )
    )
    trace = trace or TraceWriter.for_player(config.trace_dir, username)
    server_configuration = server_configuration or LocalhostServerConfiguration

    class PiAgentPlayer(Player):
        def choose_move(self_, battle: Any) -> Any:
            return _decide(
                self_,
                config=config,
                knowledge=knowledge,
                trace=trace,
                battle=battle,
            )

    account = AccountConfiguration(username, None)
    player = PiAgentPlayer(
        battle_format=battle_format,
        team=get_local_teams(battle_format, team_set_name),
        account_configuration=account,
        server_configuration=server_configuration,
        max_concurrent_battles=1,
        log_level=logging.CRITICAL,
        start_timer_on_battle_start=False,
        ping_interval=None,
        ping_timeout=None,
    )
    return player


def _decide(
    player: Any,
    *,
    config: AgentConfig,
    knowledge: KnowledgeClient | None,
    trace: TraceWriter,
    battle: Any,
) -> Any:
    """The full Pi-agent decision loop for one battle turn."""
    actions = build_actions(battle)
    if not actions:
        return player.choose_random_move(battle)
    context = TurnContext.for_turn(
        battle=battle, player=player, knowledge=knowledge if config.rag else None
    )
    prompt_text = build_prompt(battle) + "\n\nAvailable actions JSON:\n" + format_actions(actions)
    agent = build_agent(
        config, tools=build_tools(context, config), system_prompt=SYSTEM_PROMPT
    )
    result = agent.prompt_sync(prompt_text)
    answer = _assistant_text(result)
    parsed = parse_answer(answer)
    if parsed is None:
        result = agent.prompt_sync(
            "Return one legal action exactly as one JSON object. No prose, no code "
            "fences. Examples: {\"move\":\"earthquake\"} or {\"switch\":\"greattusk\"}."
        )
        answer = _assistant_text(result)
        parsed = parse_answer(answer)
    order = _resolve_order(player, context, actions, parsed)
    trace.record(
        battle=battle,
        prompt=prompt_text,
        actions=[action.name for action in actions],
        result=result,
        answer=answer or "",
        parsed=parsed,
        order=getattr(order, "label", "") or str(order),
        rag=config.rag,
    )
    return order


def _resolve_order(
    player: Any,
    context: TurnContext,
    actions: list[ActionChoice],
    parsed: dict[str, Any] | None,
) -> Any:
    if isinstance(parsed, dict):
        move_id = parsed.get("move")
        switch_id = parsed.get("switch")
        want_tera = bool(parsed.get("tera"))
        if move_id:
            found = context.find_move(str(move_id))
            if found is None:
                print(f"[{player.username}] unknown move {move_id!r}; fallback", flush=True)
            elif want_tera and found.tera:
                try:
                    return player.create_order(found.order, terastallize=True)
                except TypeError:
                    return found.order
            else:
                return found.order
        if switch_id:
            for action in actions:
                if action.kind == "switch" and action.name == switch_id:
                    return action.order
            print(f"[{player.username}] unknown switch {switch_id!r}; fallback", flush=True)
    fallback = highest_power_move(actions) or (actions[0] if actions else None)
    print(
        f"[{player.username}] fallback to {fallback.label if fallback else 'random move'} "
        f"(turn {context.battle.turn})",
        flush=True,
    )
    return fallback.order if fallback else player.choose_random_move(context.battle)


def _assistant_text(result: Any) -> str:
    for message in reversed(list(getattr(result, "messages", []) or [])):
        if type(message).__name__ != "AssistantMessage":
            continue
        pieces = [getattr(block, "text", "") for block in getattr(message, "content", [])]
        text = "\n".join(piece for piece in pieces if piece)
        if text.strip():
            return text
    return ""
