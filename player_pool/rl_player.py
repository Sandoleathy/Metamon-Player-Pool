"""Serve one metamon RL policy as an always-on local Showdown player.

The player connects, accepts challenges from anyone in ``gen9ou``, and only loads
its AMAGO checkpoint when the first challenge request arrives (unless
``--preload`` is given).  Afterwards it keeps playing battles on the same
connection until the process is stopped.

Usage:
    python -m player_pool.rl_player --agent Kakuna
    python -m player_pool.rl_player --agent Abra --preload --max-battles 1
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import sys
import threading
import time
import warnings
from typing import Optional

from player_pool.compat import (
    patch_amago_traj_length,
    patch_flash_attention,
    patch_mkdir_permissions,
    patch_torch_dynamo,
)

# must happen before torch / metamon imports
patch_mkdir_permissions()
patch_torch_dynamo()
patch_flash_attention()
patch_amago_traj_length()

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import numpy as np  # noqa: E402
from amago.envs.amago_env import SequenceWrapper  # noqa: E402
from amago.envs.env_utils import DummyAsyncVectorEnv  # noqa: E402
from poke_env import AccountConfiguration  # noqa: E402
from poke_env.concurrency import POKE_LOOP  # noqa: E402
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration  # noqa: E402

import metamon.env.wrappers as _wrappers  # noqa: E402
from metamon.env.metamon_player import PokeAgentPlayer  # noqa: E402
from metamon.env.wrappers import PokeEnvWrapper  # noqa: E402
from metamon.rl.metamon_to_amago import MetamonAMAGOWrapper  # noqa: E402
from metamon.rl.pretrained import get_pretrained_model  # noqa: E402

from player_pool.registry import FORMAT, RL_PLAYERS, TEAM_SET  # noqa: E402
from player_pool.teams import get_local_teams  # noqa: E402

# upstream metamon/env/wrappers.py forgets this import in the "pokeagent" branch
_wrappers.PokeAgentPlayer = PokeAgentPlayer

MAX_BATTLES = 10**9


class AcceptAnyChallenge(PokeEnvWrapper):
    """poke-env gym env whose player accepts challenges from any user."""

    # OpenAIGymEnv.reset gives up after `_INIT_RETRIES` polls without a battle;
    # an idle pool player must be able to wait indefinitely.
    _INIT_RETRIES = 10**9

    def __init__(self, num_battles: int = MAX_BATTLES, **kwargs):
        super().__init__(opponent_type=None, start_challenging=False, **kwargs)
        self.num_battles = num_battles
        # Run the accept loop from its own thread (the pattern poke-env expects:
        # `handle_threaded_coroutines` hops onto POKE_LOOP for us). Scheduling the
        # coroutine on POKE_LOOP directly would nest two hops and deadlock.
        self._accept_thread = threading.Thread(
            target=self._run_accept_loop, daemon=True, name=f"accept-{self.agent.username}"
        )
        self._accept_thread.start()

    def _run_accept_loop(self) -> None:
        try:
            asyncio.run(self.agent.accept_challenges(None, self.num_battles))
        except asyncio.CancelledError:
            pass
        except BaseException as exc:  # noqa: BLE001
            print(f"[{self.agent.username}] accept loop ended: {exc!r}", flush=True)


class SingleBattleVectorEnv(DummyAsyncVectorEnv):
    """One env that returns ``done`` instead of auto-resetting.

    amago's ``DummyAsyncVectorEnv`` resets finished envs inside ``step``, which
    for a ladder player means blocking until the *next* challenge arrives.  We
    want ``interact(..., episodes=1)`` to return right after a battle so the main
    loop controls resets (and can stop cleanly).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs[0].step(action[0])
        return (
            np.stack([obs], axis=0),
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated], dtype=bool),
            np.asarray([truncated], dtype=bool),
            [info],
        )


class RLPlayerRuntime:
    """Owns the player connection, the gym env and the lazily loaded policy."""

    def __init__(
        self,
        spec,
        battle_format: str,
        team_set_name: str,
        temperature: float,
        preload: bool,
        torch_threads: int,
    ):
        self.spec = spec
        self.battle_format = battle_format
        self.team_set_name = team_set_name
        self.temperature = temperature
        self.agent = None
        self._load_lock = threading.Lock()
        self._load_failed = threading.Event()
        self._battles_started = 0
        self._started_at = time.time()

        torch.set_num_threads(torch_threads)
        self.model = get_pretrained_model(spec.agent)
        self.team_set = get_local_teams(battle_format, team_set_name)

        self.env = AcceptAnyChallenge(
            battle_format=battle_format,
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            reward_function=self.model.reward_function,
            player_team_set=self.team_set,
            player_username=spec.username,
            battle_backend=self.model.battle_backend,
            turn_limit=float("inf"),
            start_timer_on_battle_start=False,
        )
        self.agent_wrapper = MetamonAMAGOWrapper(self.env)
        self.sequence = SequenceWrapper(
            self.agent_wrapper, save_trajs_to=None, save_every=None
        )
        self.vector_env = SingleBattleVectorEnv([lambda: self.sequence])

        # poke-env's PSClient keeps a bound reference to the player's challenge
        # handler, so both the player and its client must point at the lazy hook.
        lazy_handler = self._make_lazy_handler(self.env.agent._handle_challenge_request)
        self.env.agent._handle_challenge_request = lazy_handler
        self.env.agent.ps_client._handle_challenge_request = lazy_handler

        if preload:
            self.load_policy()

    # ------------------------------------------------------------------
    # policy loading
    # ------------------------------------------------------------------
    def load_policy(self) -> None:
        if self.agent is not None:
            return
        with self._load_lock:
            if self.agent is not None:
                return
            t0 = time.time()
            print(
                f"[{self.spec.username}] loading {self.spec.agent} "
                f"(epoch {self.spec.checkpoint})...",
                flush=True,
            )
            agent = self.model.initialize_agent(
                checkpoint=self.spec.checkpoint,
                log=False,
                action_temperature=self.temperature,
            )
            agent.env_mode = "sync"
            agent.parallel_actors = 1
            agent.verbose = False
            self.agent = agent
            print(
                f"[{self.spec.username}] policy ready in {time.time() - t0:.1f}s",
                flush=True,
            )

    def _make_lazy_handler(self, original):
        runtime = self

        async def handler(split_message):
            if runtime.agent is None and not runtime._load_failed.is_set():
                try:
                    runtime.load_policy()
                except Exception:
                    runtime._load_failed.set()
                    logging.exception("%s: policy load failed", runtime.spec.username)
                    return
            await original(split_message)

        return handler

    # ------------------------------------------------------------------
    # battle loop
    # ------------------------------------------------------------------
    def run(self, max_battles: int = 0) -> None:
        # "online as" must mean logged in: the smoke test and supervisors key off it,
        # and a challenge sent to a not-yet-logged-in websocket is simply lost.
        deadline = time.time() + 60
        while (
            not self.env.agent.ps_client.logged_in.is_set() and time.time() < deadline
        ):
            time.sleep(0.1)
        print(
            f"[{self.spec.username}] online as {self.spec.agent} "
            f"(format={self.battle_format}, backend={self.model.battle_backend}, "
            f"preloaded={self.agent is not None})",
            flush=True,
        )
        try:
            while True:
                self.vector_env.reset()  # blocks until the next battle starts
                if self.agent is None:  # safety net; lazy hook normally handles it
                    self.load_policy()
                self._battles_started += 1
                self.agent.interact(
                    self.vector_env, timesteps=10000, episodes=1, sample=True
                )
                agent = self.env.agent
                print(
                    f"[{self.spec.username}] battle {self._battles_started} done | "
                    f"won={agent.n_won_battles} lost={agent.n_lost_battles} "
                    f"tied={agent.n_tied_battles} | "
                    f"uptime={time.time() - self._started_at:.0f}s",
                    flush=True,
                )
                if max_battles and self._battles_started >= max_battles:
                    print(f"[{self.spec.username}] reached --max-battles", flush=True)
                    break
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.env.agent.ps_client.stop_listening(), POKE_LOOP
            )
            fut.result(timeout=5)
        except Exception:
            pass


def _spec_for(agent: str, username: Optional[str]):
    candidates = [s for s in RL_PLAYERS if s.agent == agent]
    if not candidates:
        raise SystemExit(
            f"unknown RL agent {agent!r}; available: {[s.agent for s in RL_PLAYERS]}"
        )
    spec = dataclasses.replace(candidates[0])
    if username:
        spec.username = username
    return spec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="pretrained model name")
    parser.add_argument("--username", default=None, help="Showdown username")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--format", default=FORMAT)
    parser.add_argument("--team-set", default=TEAM_SET)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--preload", action="store_true", help="load the checkpoint before the first battle"
    )
    parser.add_argument("--max-battles", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="where AMAGO runs the policy (auto = CUDA if available)",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    # accelerate reads this when the Experiment builds its Accelerator; setting it
    # here keeps a GPU machine usable for CPU-only pool members (e.g. GPU is full).
    if args.device == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"
    elif args.device == "cuda":
        os.environ.pop("ACCELERATE_USE_CPU", None)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    spec = _spec_for(args.agent, args.username)
    if args.checkpoint is not None:
        spec = dataclasses.replace(spec, checkpoint=args.checkpoint)

    runtime = RLPlayerRuntime(
        spec=spec,
        battle_format=args.format,
        team_set_name=args.team_set,
        temperature=args.temperature,
        preload=args.preload,
        torch_threads=args.torch_threads,
    )
    try:
        runtime.run(max_battles=args.max_battles)
    except KeyboardInterrupt:
        print(f"[{spec.username}] interrupted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
