"""Host every metamon heuristic baseline in a single process.

Each baseline connects to the local Showdown server under its own username and
accepts challenges from any user in the configured format.  One process is enough
because heuristic players are cheap; a crashed baseline is recreated with a new
connection (the supervisor also restarts the whole process if it dies).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings

from player_pool.compat import patch_mkdir_permissions, patch_torch_dynamo

patch_mkdir_permissions()
patch_torch_dynamo()

warnings.filterwarnings("ignore")

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from metamon.baselines import get_baseline  # noqa: E402

from player_pool.registry import FORMAT, HEURISTIC_PLAYERS, TEAM_SET  # noqa: E402
from player_pool.teams import get_local_teams  # noqa: E402

MAX_BATTLES = 10**9


def _log_result(player, battle):
    try:
        result = "WIN" if battle.won else ("TIE" if battle.tied else "LOSS")
        print(
            f"[{player.username}] {result} vs {battle.opponent_username} "
            f"({battle.battle_tag})",
            flush=True,
        )
    except Exception:
        pass


async def serve(spec, battle_format: str, team_set_name: str) -> None:
    while True:
        player = None
        try:
            cls = get_baseline(spec.baseline)
            team = get_local_teams(battle_format, team_set_name)
            player = cls(
                battle_format=battle_format,
                team=team,
                account_configuration=AccountConfiguration(spec.username, None),
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
                log_level=logging.CRITICAL,
                start_timer_on_battle_start=False,
                ping_interval=None,
                ping_timeout=None,
            )
            player._battle_finished_callback = lambda battle, p=player: _log_result(p, battle)
            print(f"[{spec.username}] online as {spec.baseline}", flush=True)
            await player.accept_challenges(None, MAX_BATTLES)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # disconnected / server restarted
            print(
                f"[{spec.username}] disconnected ({type(exc).__name__}: {exc}); retrying",
                flush=True,
            )
        finally:
            if player is not None:
                try:
                    await player.ps_client.stop_listening()
                except Exception:
                    pass
        await asyncio.sleep(5)


async def run(specs, battle_format: str, team_set_name: str) -> None:
    tasks = [asyncio.create_task(serve(s, battle_format, team_set_name)) for s in specs]
    print(f"serving {len(specs)} heuristic players in {battle_format}", flush=True)
    await asyncio.gather(*tasks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default=FORMAT)
    parser.add_argument("--team-set", default=TEAM_SET)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="subset of baseline names (default: all registered heuristics)",
    )
    args = parser.parse_args(argv)

    specs = HEURISTIC_PLAYERS
    if args.only:
        wanted = {name.lower() for name in args.only}
        specs = [
            s
            for s in specs
            if s.baseline.lower() in wanted or s.username.lower() in wanted
        ]
        if not specs:
            raise SystemExit(f"no heuristics matched {args.only}")

    try:
        asyncio.run(run(specs, args.format, args.team_set))
    except KeyboardInterrupt:
        print("heuristic server interrupted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
