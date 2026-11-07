"""A tool-using LLM player backed by ``pi-python-core`` and the local pool.

One player owns one model provider for the whole game; every turn creates a
fresh agent with that turn's tools, awaits the model, and closes the agent.  A
turn with a single legal action is answered without spending a model call, and
every failure path returns a legal order instead of raising into the battle
loop.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from .config import AgentConfig, SYSTEM_PROMPT, agent_config_from_env
from .knowledge import KnowledgeClient
from .prompt import (
    ActionChoice,
    build_actions,
    build_prompt,
    build_repair_prompt,
    format_actions,
    highest_power_move,
    parse_answer,
)
from .provider import build_agent, opencode_provider
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

    The model call is awaited inside poke-env's battle loop, so the websocket
    keeps being serviced while the agent thinks.  The battle timer stays
    disabled to keep long tool-driven turns from forfeiting the game.
    """
    # Import order matters: metamon.baselines builds its own GenData instances at
    # import time, and poke-env's GenData refuses a second construction for a gen
    # it has already registered.  Importing this package after a battle started
    # therefore raises "GenData for gen 9 already initialized" and takes the
    # damage tool down with it, so load it before the first Battle exists.
    try:
        import metamon.baselines  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — damage tool degrades, battle continues
        print(
            f"pi_agent: metamon.baselines unavailable ({type(exc).__name__}: {exc}); "
            f"calculate_damage will report unavailable",
            flush=True,
        )

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
    provider = opencode_provider(config)

    class PiAgentPlayer(Player):
        async def choose_move(self_, battle: Any) -> Any:
            forced = _forced_order(self_, battle)
            if forced is not None:
                trace.record_forced(battle, forced, reason="single legal action")
                return forced
            return await _decide(
                self_,
                config=config,
                provider=provider,
                knowledge=knowledge,
                trace=trace,
                battle=battle,
            )

        async def close(self_) -> None:
            """Release the shared model provider; safe to call more than once."""
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                pass

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


def _forced_order(player: Any, battle: Any) -> Any | None:
    """The only legal order, when there is exactly one and no choice matters."""
    active = getattr(battle, "active_pokemon", None)
    moves = list(getattr(battle, "available_moves", None) or [])
    switches = list(getattr(battle, "available_switches", None) or [])
    if active is not None and getattr(active, "fainted", False) and len(switches) == 1:
        return player.create_order(switches[0])
    if len(moves) == 1 and not switches:
        return player.create_order(moves[0])
    if not moves and len(switches) == 1:
        return player.create_order(switches[0])
    return None


def _label(order: Any) -> str:
    if order is None:
        return ""
    return getattr(order, "message", None) or str(order)


async def _decide(
    player: Any,
    *,
    config: AgentConfig,
    provider: Any,
    knowledge: KnowledgeClient | None,
    trace: TraceWriter,
    battle: Any,
) -> Any:
    """The full Pi-agent decision loop for one battle turn."""
    actions = build_actions(battle)
    if not actions:
        trace.record_forced(battle, None, reason="no legal actions")
        return player.choose_random_move(battle)

    context = TurnContext.for_turn(
        battle=battle, player=player, knowledge=knowledge if config.rag else None
    )
    prompt_text = build_prompt(battle) + "\n\nAvailable actions JSON:\n" + format_actions(actions)
    agent = build_agent(
        config, tools=build_tools(context, config), system_prompt=SYSTEM_PROMPT, provider=provider
    )
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    order: Any | None = None
    result: Any = None
    answer = ""
    parsed: dict[str, Any] | None = None

    try:
        result = await agent.prompt(prompt_text)
        answer = _assistant_text(result)
        parsed = parse_answer(answer)
        order = _resolve_order(player, context, actions, parsed)
        attempts.append({"kind": "action", "answer": answer, "parsed": parsed, "chosen": _label(order)})

        if order is None:
            if not answer.strip():
                # An empty answer means the call itself failed (bad key, dead
                # endpoint).  A repair prompt would only burn another request.
                attempts.append({"kind": "skip_repair", "reason": "empty model answer"})
            else:
                repair_prompt = build_repair_prompt(actions, answer)
                result = await agent.prompt(repair_prompt)
                repaired = _assistant_text(result)
                repaired_parsed = parse_answer(repaired)
                repaired_order = _resolve_order(player, context, actions, repaired_parsed)
                attempts.append(
                    {
                        "kind": "repair",
                        "answer": repaired,
                        "parsed": repaired_parsed,
                        "chosen": _label(repaired_order),
                    }
                )
                if repaired_order is not None:
                    order, answer, parsed = repaired_order, repaired, repaired_parsed
    except Exception as exc:  # noqa: BLE001 — a failed call must fall back, not crash
        attempts.append({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        print(
            f"[{player.username}] model call failed ({type(exc).__name__}: {exc}); using fallback",
            flush=True,
        )
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001 — cleanup must not raise
            pass

    if order is None:
        fallback = highest_power_move(actions) or actions[0]
        print(
            f"[{player.username}] fallback to {fallback.label} (turn {getattr(battle, 'turn', '-')})",
            flush=True,
        )
        try:
            order = player.create_order(fallback.order)
        except Exception:  # noqa: BLE001 — last resort before random
            order = player.choose_random_move(battle)

    trace.record(
        battle=battle,
        prompt=prompt_text,
        actions=[action.name for action in actions],
        result=result,
        answer=answer,
        parsed=parsed,
        order=_label(order),
        rag=config.rag,
        attempts=attempts,
        latency=time.monotonic() - started,
    )
    return order


def _resolve_order(
    player: Any,
    context: TurnContext,
    actions: list[ActionChoice],
    parsed: dict[str, Any] | None,
) -> Any | None:
    """Map a parsed answer to a legal order, or ``None`` when it is not legal."""
    if not isinstance(parsed, dict):
        return None

    move_id = parsed.get("move")
    switch_id = parsed.get("switch")
    want_tera = bool(parsed.get("tera") or parsed.get("terastallize") is True)
    action_hint = str(parsed.get("action") or "").strip().lower()
    if not move_id and not switch_id and parsed.get("target"):
        if action_hint == "switch":
            switch_id = parsed.get("target")
        else:
            move_id = parsed.get("target")

    if move_id:
        found = context.find_move(str(move_id))
        if found is None:
            return None
        if want_tera and found.tera:
            try:
                return player.create_order(found.order, terastallize=True)
            except TypeError:
                return player.create_order(found.order)
        return player.create_order(found.order)

    if switch_id:
        for action in actions:
            if action.kind == "switch" and action.name == switch_id:
                return player.create_order(action.order)
    return None


def _assistant_text(result: Any) -> str:
    for message in reversed(list(getattr(result, "messages", []) or [])):
        if type(message).__name__ != "AssistantMessage":
            continue
        pieces = [getattr(block, "text", "") for block in getattr(message, "content", [])]
        text = "\n".join(piece for piece in pieces if piece)
        if text.strip():
            return text
    return ""
