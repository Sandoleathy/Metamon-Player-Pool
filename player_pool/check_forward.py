"""Device-independent forward-pass check for a pretrained policy.

Loads one model exactly like ``rl_player`` does, then runs a few steps of
``policy.get_actions`` on synthetic observations.  This exercises the tstep
encoder and the trajectory encoder (including the FlashAttention / KV-cache
path) without needing a Showdown server, and works on CPU or CUDA.

Usage:
    python -m player_pool.check_forward --agent Kakuna
    python -m player_pool.check_forward --agent Abra --steps 5
"""
from __future__ import annotations

import argparse
import sys

from player_pool.compat import (
    patch_amago_traj_length,
    patch_flash_attention,
    patch_mkdir_permissions,
    patch_torch_dynamo,
)

patch_mkdir_permissions()
patch_torch_dynamo()
patch_flash_attention()
patch_amago_traj_length()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from player_pool.registry import RL_PLAYERS  # noqa: E402
from metamon.rl.pretrained import get_pretrained_model  # noqa: E402


def _dummy_obs(model, device: str):
    space = model.observation_space.gym_space
    obs = {}
    for key, sub in space.spaces.items():
        if np.issubdtype(sub.dtype, np.integer):
            dtype = torch.long
        elif np.issubdtype(sub.dtype, np.bool_):
            dtype = torch.bool
        else:
            dtype = torch.float32
        obs[key] = torch.zeros((1, 1) + tuple(sub.shape), dtype=dtype, device=device)

    action_dim = model.action_space.gym_space.n
    obs["illegal_actions"] = torch.zeros((1, 1, action_dim), dtype=torch.bool, device=device)
    rl2s = torch.zeros((1, 1, action_dim + 1), dtype=torch.float32, device=device)
    # amago's traj encoder expects a trailing feature dim: (batch, time, 1)
    time_idxs = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    return obs, rl2s, time_idxs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="Kakuna")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)

    spec = next((s for s in RL_PLAYERS if s.agent == args.agent), None)
    if spec is None:
        raise SystemExit(f"unknown agent {args.agent!r}")
    checkpoint = args.checkpoint if args.checkpoint is not None else spec.checkpoint

    model = get_pretrained_model(spec.agent)
    agent = model.initialize_agent(
        checkpoint=checkpoint, log=False, action_temperature=args.temperature
    )
    policy = agent.policy
    policy.eval()
    device = agent.DEVICE

    obs, rl2s, time_idxs = _dummy_obs(model, device)
    hidden = policy.traj_encoder.init_hidden_state(1, device)

    print(
        f"{spec.agent} (epoch {checkpoint}) on {device}: "
        f"obs keys={sorted(obs)} action_dim={model.action_space.gym_space.n}",
        flush=True,
    )
    with torch.no_grad():
        for step in range(args.steps):
            actions, hidden = policy.get_actions(
                obs=obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                sample=True,
                hidden_state=hidden,
            )
            print(f"  step {step}: action={actions.squeeze(1).tolist()}", flush=True)
            time_idxs = time_idxs + 1
    print("forward pass OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
