"""Smoke-test the pool: spawn each RL player and play one battle against it.

Usage:
    python -m player_pool.smoke_test                 # all RL models, 1 battle each
    python -m player_pool.smoke_test --models Abra Kakuna
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

from player_pool.compat import patch_mkdir_permissions, patch_torch_dynamo

patch_mkdir_permissions()
patch_torch_dynamo()
warnings.filterwarnings("ignore")

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

from player_pool.registry import FORMAT, RL_PLAYERS  # noqa: E402
from player_pool.teams import get_local_teams  # noqa: E402

LOG_DIR = Path(__file__).parent / "logs"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _tail_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


async def _battle(username: str, fmt: str) -> str:
    team = get_local_teams(fmt, "competitive")
    challenger = "chal" + "".join(str(random.randint(0, 9)) for _ in range(6))
    player = RandomPlayer(
        battle_format=fmt,
        team=team,
        account_configuration=AccountConfiguration(challenger, None),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
        log_level=logging.CRITICAL,
        ping_interval=None,
        ping_timeout=None,
    )
    try:
        await asyncio.wait_for(player.ps_client.logged_in.wait(), timeout=30)
        await asyncio.wait_for(player.send_challenges(username, 1), timeout=300)
        return f"won={player.n_won_battles} lost={player.n_lost_battles}"
    except Exception as exc:  # noqa: BLE001
        return f"FAILED {type(exc).__name__}: {exc}"
    finally:
        try:
            await player.ps_client.stop_listening()
        except Exception:
            pass


def run_model(spec) -> str:
    LOG_DIR.mkdir(exist_ok=True)
    username = f"SM-{spec.display_name}"[:18]
    log_path = LOG_DIR / f"smoke_{spec.agent}.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "player_pool.rl_player",
            "--agent",
            spec.agent,
            "--username",
            username,
            "--max-battles",
            "1",
        ],
        cwd=str(REPO_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if _tail_contains(log_path, "online as"):
                break
            if proc.poll() is not None:
                return f"PLAYER EXITED (code {proc.returncode}); see {log_path.name}"
            time.sleep(1.0)
        else:
            return f"TIMEOUT waiting for login; see {log_path.name}"
        return asyncio.run(_battle(username, FORMAT))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args(argv)

    specs = RL_PLAYERS
    if args.models:
        wanted = {m.lower() for m in args.models}
        specs = [s for s in specs if s.agent.lower() in wanted]
        if not specs:
            raise SystemExit(f"no models matched {args.models}")

    failures = 0
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.agent} ({spec.username}) ...", flush=True)
        t0 = time.time()
        result = run_model(spec)
        print(f"    {result}  [{time.time() - t0:.0f}s]", flush=True)
        if result.startswith(("FAILED", "PLAYER", "TIMEOUT")):
            failures += 1
    print(f"\n{len(specs) - failures}/{len(specs)} models played a battle")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
