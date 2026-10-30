"""Supervisor for the local Gen9 OU opponent pool.

Starts one process per metamon RL policy plus one process hosting all heuristic
baselines, restarts anything that dies, and writes:

* ``player_pool/players.json`` — the registry an LLM agent reads to pick an
  opponent (username, format, strength notes, ready-made challenge command).
* ``player_pool/status.json``  — which pool processes are currently alive.

Usage:
    python -m player_pool.pool                 # everything, forever
    python -m player_pool.pool --dry-run       # print the commands only
    python -m player_pool.pool --rl-only       # no heuristic process
    python -m player_pool.pool --only-rl Kakuna Abra
    python -m player_pool.pool --only-heuristic Grunt RandomBaseline
    python -m player_pool.pool --preload Kakuna Abra Kadabra3
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from player_pool.registry import (
    FORMAT,
    HEURISTIC_PLAYERS,
    PLAYERS_JSON,
    RL_PLAYERS,
    TEAM_SET,
    PlayerSpec,
    write_players_json,
)

POOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = POOL_DIR.parent
LOG_DIR = POOL_DIR / "logs"
STATUS_JSON = POOL_DIR / "status.json"

RESTART_DELAY = 5.0
MAX_BACKOFF = 60.0
MONITOR_INTERVAL = 2.0
HEALTHY_UPTIME = 60.0


def _python() -> str:
    return sys.executable


def select_players(args) -> tuple[list[PlayerSpec], list[PlayerSpec]]:
    """Return (rl_specs, heuristic_specs) that should be launched."""
    rl_specs = list(RL_PLAYERS)
    if args.only_rl:
        wanted = {name.lower() for name in args.only_rl}
        rl_specs = [
            s
            for s in rl_specs
            if s.agent.lower() in wanted or s.username.lower() in wanted
        ]
        if not rl_specs:
            raise SystemExit(f"no RL agents matched {args.only_rl}")

    heuristic_specs = (
        [] if (args.no_heuristics or args.rl_only) else list(HEURISTIC_PLAYERS)
    )
    if args.only_heuristic:
        wanted = {name.lower() for name in args.only_heuristic}
        heuristic_specs = [
            s
            for s in heuristic_specs
            if s.baseline.lower() in wanted or s.username.lower() in wanted
        ]
        if not heuristic_specs and not args.no_heuristics:
            raise SystemExit(f"no heuristics matched {args.only_heuristic}")
    return rl_specs, heuristic_specs


def build_commands(
    args, rl_specs: list[PlayerSpec], heuristic_specs: list[PlayerSpec]
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []

    if heuristic_specs:
        cmd = [
            _python(),
            "-u",
            "-m",
            "player_pool.heuristic_server",
            "--format",
            args.format,
            "--team-set",
            args.team_set,
            "--only",
            *[s.baseline for s in heuristic_specs],
        ]
        commands.append(("heuristics", cmd))

    preload = {name.lower() for name in args.preload}
    for spec in rl_specs:
        cmd = [
            _python(),
            "-u",
            "-m",
            "player_pool.rl_player",
            "--agent",
            spec.agent,
            "--username",
            spec.username,
            "--checkpoint",
            str(spec.checkpoint),
            "--format",
            args.format,
            "--team-set",
            args.team_set,
            "--torch-threads",
            str(args.torch_threads),
            "--device",
            args.device,
        ]
        if spec.agent.lower() in preload or spec.username.lower() in preload:
            cmd.append("--preload")
        commands.append((spec.username, cmd))

    return commands


def _start(name: str, cmd: list[str], log_dir: Path) -> subprocess.Popen:
    log_path = log_dir / f"{name.replace('-', '_')}.log"
    log_file = open(log_path, "ab", buffering=0)
    header = (
        f"\n===== {datetime.now().isoformat(timespec='seconds')} start: "
        f"{' '.join(cmd)}\n"
    )
    log_file.write(header.encode())
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    log_file.close()
    return proc


def _terminate(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _write_status(procs: dict[str, subprocess.Popen], attempts: dict[str, int]) -> None:
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "processes": {
            name: {
                "state": "running" if proc.poll() is None else "down",
                "pid": proc.pid,
                "restarts": attempts.get(name, 0),
            }
            for name, proc in procs.items()
        },
    }
    try:
        STATUS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default=FORMAT)
    parser.add_argument("--team-set", default=TEAM_SET)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="device for the RL policies (auto = CUDA if available)",
    )
    parser.add_argument("--rl-only", action="store_true")
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument(
        "--only-rl", nargs="*", default=None, help="subset of RL agents to run"
    )
    parser.add_argument(
        "--only-heuristic", nargs="*", default=None, help="subset of heuristic baselines"
    )
    parser.add_argument(
        "--preload",
        nargs="*",
        default=[],
        help="RL agents that load their checkpoint at startup instead of on first challenge",
    )
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--duration", type=int, default=0, help="stop after N seconds (0 = forever)"
    )
    args = parser.parse_args(argv)

    rl_specs, heuristic_specs = select_players(args)
    commands = build_commands(args, rl_specs, heuristic_specs)
    launched = rl_specs + heuristic_specs
    players_path = write_players_json(battle_format=args.format, specs=launched)
    print(
        f"wrote {players_path} ({len(launched)} players: "
        f"{len(rl_specs)} RL, {len(heuristic_specs)} heuristic)",
        flush=True,
    )

    if args.dry_run:
        print(f"\n{len(commands)} processes:")
        for name, cmd in commands:
            print(f"  {name}: {' '.join(cmd)}")
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"starting {len(commands)} processes; logs in {log_dir}", flush=True)

    procs: dict[str, subprocess.Popen] = {}
    attempts: dict[str, int] = {}
    retry_at: dict[str, float] = {}
    started_at: dict[str, float] = {}
    stopping = False

    def handle_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, handle_stop)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_stop)

    t0 = time.time()
    last_status = 0.0
    try:
        while not stopping:
            now = time.time()
            for name, cmd in commands:
                proc = procs.get(name)
                if proc is not None and proc.poll() is None:
                    continue
                if proc is not None:
                    code = proc.returncode
                    uptime = now - started_at.get(name, now)
                    if uptime > HEALTHY_UPTIME:
                        attempts[name] = 0
                    attempts[name] = attempts.get(name, 0) + 1
                    delay = min(
                        RESTART_DELAY * (2 ** (attempts[name] - 1)), MAX_BACKOFF
                    )
                    retry_at[name] = now + delay
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] {name} exited "
                        f"after {uptime:.0f}s (code {code}); restarting in {delay:.0f}s",
                        flush=True,
                    )
                    procs[name] = None
                    continue
                if retry_at.get(name, 0) > now:
                    continue
                procs[name] = _start(name, cmd, log_dir)
                started_at[name] = time.time()
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] started {name} "
                    f"(pid {procs[name].pid})",
                    flush=True,
                )
            if now - last_status > 5:
                _write_status(
                    {name: p for name, p in procs.items() if p is not None}, attempts
                )
                last_status = now
            if args.duration and time.time() - t0 > args.duration:
                print("duration reached; stopping", flush=True)
                break
            time.sleep(MONITOR_INTERVAL)
    finally:
        print("stopping players...", flush=True)
        for proc in procs.values():
            if proc is not None:
                _terminate(proc)
        print("all players stopped", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
