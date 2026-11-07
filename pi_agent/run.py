"""Challenge a registered pool player with the tool-using Pi LLM agent.

Example:
    OPENCODE_API_KEY=oc_sk_...
    OPENCODE_MODEL=deepseek-v4.1-flash
    python -m pi_agent.run --opponent MM-Kakuna --rag on/off
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .player import create_player


async def play_one(
    username: str,
    *,
    opponent: str,
    rag: bool,
    model: str | None,
    thinking: str | None,
    trace_dir: Path,
) -> str:
    """Send one challenge and report the win/loss/tie summary."""
    os.environ["PI_AGENT_RAG"] = "1" if rag else "0"
    if thinking:
        os.environ["PI_AGENT_THINKING"] = thinking

    from .config import agent_config_from_env

    config = agent_config_from_env(rag=rag, model=model, trace_dir=trace_dir)
    player = create_player(username=username, config=config)
    try:
        await asyncio.wait_for(player.ps_client.logged_in.wait(), timeout=60)
        print(f"[{username}] online", flush=True)
        await asyncio.wait_for(player.send_challenges(opponent, 1), timeout=300)
        return (
            f"won={getattr(player, 'n_won_battles', 0)} "
            f"lost={getattr(player, 'n_lost_battles', 0)} "
            f"tied={getattr(player, 'n_tied_battles', 0)}"
        )
    finally:
        try:
            await player.close()
        except Exception:
            pass
        try:
            await player.ps_client.stop_listening()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", required=True, help="pool username, e.g. MM-Kakuna")
    parser.add_argument("--username", default="pi-agent-agent")
    parser.add_argument("--rag", choices=["on", "off"], default=os.environ.get("PI_AGENT_RAG", "on"))
    parser.add_argument("--model", default=None, help="override OPENCODE_MODEL")
    parser.add_argument("--thinking", default=None)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(os.environ.get("PI_AGENT_TRACE_DIR", "pi_agent/logs")),
    )
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(
            play_one(
                username=args.username[:18],
                opponent=args.opponent,
                rag=args.rag.startswith("on"),
                model=args.model,
                thinking=args.thinking,
                trace_dir=args.trace_dir,
            )
        )
    except KeyboardInterrupt:
        result = "interrupted"
    print(f"RESULT {result}", flush=True)
    return 0


__all__ = ["main", "play_one"]

if __name__ == "__main__":
    raise SystemExit(main())
