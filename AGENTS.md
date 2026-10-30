# AGENTS.md

This checkout is upstream **metamon v1.6.0** plus a local **Gen 9 OU opponent
pool** (`player_pool/`) and a small set of environment fixes. This file is the
change record for coding agents: what was added or modified, why, what contracts
it creates, and what must not be undone. Upstream usage documentation lives in
[README.md](README.md); pool usage lives in
[player_pool/README.md](player_pool/README.md).

Read this before changing `player_pool/`, reverting local edits, or touching the
Showdown-server assumptions. If you need a server to run tests against, start
with [Before you start](#before-you-start-clone-and-run-the-showdown-server).

## Before you start: clone and run the Showdown server

The pool **does not start, stop or bundle a server**, and the
`server/pokemon-showdown` submodule is empty in this checkout. Every developer
must clone the Showdown server themselves, start it locally, and only then run
the pool/tests against it.

1. Get the pinned server (this is upstream `smogon/pokemon-showdown`, the same
   repo this checkout's submodule points at):

   ```shell
   # option A: let git fill the existing submodule (checks out the pinned commit)
   git submodule update --init server/pokemon-showdown

   # option B: clone it yourself and check out the same pin
   git clone https://github.com/smogon/pokemon-showdown.git server/pokemon-showdown
   cd server/pokemon-showdown
   git checkout d62d3a39878f7ef1c6e1cfbd84e8feb2ce7a9fdd
   ```

   The pin is the commit recorded in this repo's index
   (`git ls-tree HEAD server/pokemon-showdown`). It matters: metamon
   re-simulates battles client-side, and newer server data can desync the
   policies. Do not commit the cloned server; the submodule pointer is the
   version-controlled part.

2. Install dependencies and start the server (Node 18+):

   ```shell
   cd server/pokemon-showdown
   npm install
   node pokemon-showdown start --no-security
   ```

   - `--no-security` is mandatory; without it bots fail to log in with
     `|nametaken|...|Your authentication token was invalid.`
   - The default port is **8000**, which is what
     `poke_env.LocalhostServerConfiguration` and therefore `player_pool` expects.
     Only one process can own it: stop any other Showdown instance first.
   - Keep the server running in its own terminal (or a service manager). The
     pool and test scripts assume it is already up; they never launch it.
   - Windows: if PowerShell blocks `npm.ps1`, run `npm.cmd install`; the
     installed `node` must be on `PATH`.

3. Verify the server before blaming the pool:

   ```powershell
   Test-NetConnection 127.0.0.1 -Port 8000      # or any TCP probe
   python -m player_pool.smoke_test --models Minikazam
   ```

   A healthy run plays one battle in ≈1 min. Common failures:
   `Authentication token was invalid` → server missing `--no-security`;
   `Your team was rejected ...` → the server's dex/rules differ from the team set
   (see §3.5); `Unrecognized format` → use `gen9ou` (or add the format you pass
   via `--format` to the server).

4. Only then run the full pool (`python -m player_pool.pool`) and the rest of the
   verification ladder.

Do **not** point the pool at the public `play.pokemonshowdown.com`: it requires
registered accounts and is not intended for bot pools. If the LLM agent needs a
non-stock format (e.g. a long-timer variant), add it in
`server/pokemon-showdown/config/custom-formats.ts`, restart the server, and pass
`--format <new id>` to the pool.

## 1. Working tree: what is ours

| path | git state | what it is |
|---|---|---|
| `player_pool/` | **new, untracked** | ~1.5k lines of Python + docs: the 26-player opponent pool and its tooling |
| `AGENTS.md` | **new, untracked** | this document |
| `.gitignore` | modified (this session) | now ignores the 3.2 GB `metamon_cache/` |
| `server/pokemon-showdown` | submodule, **empty** | pinned Showdown server; clone and run it yourself ([Before you start](#before-you-start-clone-and-run-the-showdown-server)) |
| `metamon/config.py` | modified (pre-existing) | cache-dir default |
| `pyproject.toml` | modified (pre-existing) | `jupyter` + `python-dotenv` dependencies |
| `metamon/rl/configs/__init__.py` | modified (pre-existing) | unused cache constant |
| `metamon_cache/` | ignored runtime data | checkpoints (~2.6 GB) + team sets + usage stats (~3.2 GB total) |
| `.venv/`, `uv.lock`, `baseline_test.ipynb` | untracked, pre-existing | local environment, not part of this change |

Sections 3.1–3.9 were added in this collaboration. The root `.gitignore` change
(ignoring `metamon_cache/`) was also made in this session. Section 3.10 documents
tracked-file edits that already existed in the working tree — they are part of
the local setup but were not authored by the pool work.

**No file under `metamon/` was modified by the pool work itself.** Keep it that
way: new behavior belongs in `player_pool/`, so upstream stays easy to rebase.

## 2. Why these changes exist

The PokéAgent Challenge Battling Track evaluates LLM agents against a pool of
metamon baselines on Gen9OU. To reproduce that setup against a locally hosted
Showdown server (so our own LLM agent can challenge the same opponents), we
needed the Gen9OU-capable metamon agents to exist as ordinary Showdown players.

Constraints discovered while building it:

- the server must be started with `--no-security`, otherwise bots cannot log in;
- the pool targets `gen9ou`; the stock pinned server has other formats, but our
  teams and the `players.json` contract only cover `gen9ou`;
- the machine is CPU-only, while 8 of 10 policies use CUDA-only FlashAttention
  and amago networks are decorated with `torch.compile`;
- upstream helpers (`Player.battle_against`, `metamon.rl.evaluate`) do not run
  unmodified with this poke-env/AMAGO combination;
- ten checkpoints total ~2.8 GB and must not all be resident in RAM while idle.

Result: 26 Showdown accounts (10 RL policies + 16 heuristic/IL baselines) that
idle on `localhost:8000`, accept challenges from anyone in `gen9ou`, and are
described to the LLM in `player_pool/players.json`.

## 3. Change inventory

### 3.1 `player_pool/registry.py` — 26 player specs, one JSON contract

The single source of truth for usernames, checkpoints and player metadata.
`PlayerSpec` fields: `username`, `display_name`, `kind`, `description`, `agent`
or `baseline`, `checkpoint`, `params`, `strength`, `tags`, plus a derived
`format` and `challenge` string.

RL players (all trained for Gen 9 OU; epochs match `metamon/rl/pretrained.py`
defaults):

| username | model | ckpt | params |
|---|---|---|---|
| `MM-Kakuna` | Kakuna | 34 | 142M |
| `MM-Kadabra3` | Kadabra3 | 20 | 57M |
| `MM-Kadabra4` | Kadabra4 | 50 | 57M |
| `MM-Alakazam` | Alakazam | 8 | 57M |
| `MM-Abra` | Abra | 40 | 57M |
| `MM-Superkazam` | Superkazam | 50 | 142M |
| `MM-Kadabra2` | Kadabra2 | 44 | 57M |
| `MM-Kadabra` | Kadabra | 46 | 57M |
| `MM-Minikazam` | Minikazam | 40 | 4.7M |
| `MM-SmallRLGen9` | SmallRLGen9Beta | 24 | ~15M |

Heuristic/IL players: `MM-Random`, `MM-BugCatcher`, `MM-MaxBP`,
`MM-Gen1Trainer`, `MM-Gen1Switch`, `MM-Gen1Boss`, `MM-BasicSwitch`, `MM-Grunt`,
`MM-GymLeader`, `MM-SmogonSwitch`, `MM-BaseRNN`, `MM-WinsOnlyRNN`, `MM-MiniRNN`,
`MM-KaizoPlus`, `MM-EmeraldKaizo`, `MM-PokeEnv`.

`write_players_json()` emits `players.json`; the `strength` notes are copied
from the paper/docstrings so the LLM can sample by strength instead of uniformly.

### 3.2 `player_pool/rl_player.py` — one checkpoint as an always-on player

Wraps a metamon policy in the same gym/AMAGO stack used by
`metamon.rl.evaluate`, but keeps the account online indefinitely instead of
running a fixed batch:

- `AcceptAnyChallenge` (a `PokeEnvWrapper` subclass) sets
  `_INIT_RETRIES = 10**9` and runs `accept_challenges(None, ...)` from its **own
  thread** (calling it on `POKE_LOOP` deadlocks).
- **Lazy load**: `_handle_challenge_request` is wrapped on both the player and
  `player.ps_client` (poke-env stores a bound reference at init). The checkpoint
  is loaded *before* the challenge is accepted, so idle players cost no model
  RAM and no battle timer runs during the load.
- `SingleBattleVectorEnv` replaces amago's `DummyAsyncVectorEnv` so finishing a
  battle does not auto-reset into a battle that does not exist yet.
- The CLI supports `--agent`, `--username`, `--checkpoint`, `--format`,
  `--team-set`, `--temperature`, `--preload`, `--max-battles`,
  `--torch-threads`, `--device {auto,cpu,cuda}` and `--log-level`.
- It injects `PokeAgentPlayer` into `metamon.env.wrappers` because the upstream
  module references it but forgets to import it.
- Prints one flushed line per battle (`won/lost/tied`, uptime) to its log.

### 3.3 `player_pool/heuristic_server.py` — all 16 baselines in one process

One asyncio process keeps 16 independent `poke_env.Player` connections alive.
Each player runs `accept_challenges(None, ...)`; if a connection dies it is
recreated after 5 s. Battle results are logged through
`_battle_finished_callback`. `--only` limits the set (used by the supervisor's
subset flags).

### 3.4 `player_pool/pool.py` — supervisor

Builds and runs the process tree: **one heuristic process + one process per RL
policy** (11 by default). It writes `players.json` for the launched subset,
starts children with the current interpreter (`sys.executable`) and per-process
logs under `player_pool/logs/`, monitors them, restarts crashes with exponential
backoff (5 s → 60 s, reset after 60 s of healthy uptime), writes `status.json`
every 5 s, and terminates everything on Ctrl+C or `--duration`.

Flags: `--only-rl`, `--only-heuristic`, `--rl-only`, `--no-heuristics`,
`--preload`, `--device`, `--torch-threads`, `--format`, `--team-set`,
`--log-dir`, `--duration`, `--dry-run`.

### 3.5 `player_pool/teams.py` — teams the local server accepts

`LocalServerTeamSet(TeamSet)` rejects packed teams containing `terablast`,
`zoroark`, `zorua` or `revivalblessing`. The server rejects Tera Blast at team
preview; without the filter the battle never starts and the RL player reports
`Agent is not challenging`. `get_local_teams()` wraps `get_metamon_teams` with
this class. If the server ruleset changes, update the substring list.

### 3.6 `player_pool/compat.py` — four environment patches

These fix failures that otherwise make the pool unusable. They are applied by
every entry point before importing torch/metamon/amago.

| patch | problem it fixes | effect | active when / override |
|---|---|---|---|
| `patch_mkdir_permissions` | Python 3.14 on Windows turns `os.mkdir(..., 0o700)` into a restrictive DACL; `tempfile.mkdtemp` dirs become unwritable and every `huggingface_hub` download fails | forces `0o777` for 0o700 mkdir calls | Windows only; no-op elsewhere |
| `patch_torch_dynamo` | amago's `@torch.compile` triggers Inductor, which needs a C++ toolchain and piped compiler subprocesses (unavailable here) | disables dynamo and sets `TORCHDYNAMO_DISABLE` | Windows or no CUDA by default; `PLAYER_POOL_TORCH_COMPILE=0/1` |
| `patch_flash_attention` | 8/10 policies use `amago.nets.transformer.FlashAttention`; flash-attn 2 is CUDA-only | plain-PyTorch attention with the same causal + sliding-window mask and KV cache, sharing the original weights | only when native flash-attn cannot run; `PLAYER_POOL_FLASH_ATTENTION=native/cpu` |
| `patch_amago_traj_length` | amago's default `traj_save_len = 1e10` is a float, rejected by `random.randint` in `Experiment.start()` | rounds the value to int | always |

The CUDA behavior is deliberate: with CUDA + flash-attn installed, the upstream
fast path and `torch.compile` stay enabled, so the project remains portable to a
GPU machine. Do not make these patches unconditional again.

### 3.7 Generated artifacts

- `player_pool/players.json` — the LLM-facing registry: `username`, `format`,
  `kind`, `agent`/`baseline`, `checkpoint`, `params`, `strength`, `tags`,
  `challenge` (`/challenge MM-Kakuna, gen9ou`). Written by `pool.py` on start and
  by `python -m player_pool.registry`. Do not hand-edit; change the registry.
- `player_pool/status.json` — last-known state/pid/restart count per process,
  rewritten every 5 s by the supervisor. It is a liveness hint, not proof that a
  player is logged in, and goes stale after shutdown.
- `player_pool/logs/*.log` — one file per process; gitignored.

### 3.8 Tooling added with the pool

- `download_models.py` — prefetches the 10 checkpoints into the HF cache
  (`metamon_cache/pretrained_models`), one `get_path_to_checkpoint` call each.
- `smoke_test.py` — spawns one RL player at a time, waits for `online as`,
  challenges it with a `RandomPlayer`, reports win/loss (needs the server).
- `check_forward.py` — loads a checkpoint and runs `policy.get_actions` on
  synthetic observations; verifies encoders, the attention path and the KV cache
  without a server. Device-aware, so use it first on a new machine.
- `check_compat.py` — six one-process cases that assert the patch-selection
  logic (CPU, simulated CUDA with/without flash-attn, forced modes).

### 3.9 Docs and ignore rules

- `player_pool/README.md` — usage, architecture, CPU/CUDA behavior, tests.
- `player_pool/.gitignore` — `logs/`, `status.json`, `__pycache__/`.
- root `.gitignore` — added `metamon_cache/` (the ~3.2 GB runtime cache) so a
  broad `git add .` cannot commit downloaded checkpoints or usage stats.

### 3.10 Pre-existing edits to tracked upstream files

These were in the working tree before the pool work. They are intentional; do
not revert them without asking the user.

```diff
# metamon/config.py
- METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", None)
+ METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", './metamon_cache')
```

Purpose: `metamon` works without exporting `METAMON_CACHE_DIR` (matches `.env`).
Risk: `metamon.rl.pretrained` raises when the value is `None`, so reverting this
breaks runs that rely on the default.

```diff
# pyproject.toml
   "amago @ git+https://github.com/UT-Austin-RPL/amago@v3.4.0",
+  "jupyter>=1.1.1",
+  "python-dotenv>=1.2.4",
```

Purpose: `baseline_test.ipynb` uses `load_dotenv()` and Jupyter; the pool runtime
does not need either. Harmless for the pool, but it changes the lock/dependency
set.

```diff
# metamon/rl/configs/__init__.py (new content in an empty file)
+import os
+METAMON_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".metamon_cache")
```

Purpose: none found — nothing imports this constant, and it contradicts
`metamon/config.py`. Treat it as dead code; remove only with user approval.

### 3.11 Downloaded data (`metamon_cache/`, gitignored)

Everything here is generated at runtime and fully regenerable; it is ignored by
the root `.gitignore` as of this session.

- 10 policy checkpoints, ~2.58 GB, under
  `pretrained_models/models--jakegrigsby--metamon/`.
- `teams/competitive/gen9ou` — 16 sample teams (cache version v5), downloaded
  through `get_metamon_teams`.
- `usage-stats/` — created on demand by some heuristics (e.g. KaizoPlus).

Refill with `python -m player_pool.download_models`. Do not delete this
directory without asking; a re-download is multi-GB.

## 4. Runtime topology

```
LLM agent ── reads players.json ──┐
        └── /challenge MM-X, gen9ou ──▶ Showdown :8000 ◀── websockets ── pool
                                          ▲                              ├─ heuristics process (16 players)
                                          └──────────────────────────────└─ 10 × rl_player (lazy policy load)
```

- Players never queue on the ladder; they accept direct challenges only. This
  avoids baselines fighting each other and keeps compute for the LLM agent.
- One battle at a time per player (`max_concurrent_battles=1`); extra challenges
  queue inside poke-env.
- Idle RL processes use a few hundred MB each; the policy loads on the first
  challenge (1–6 s here). `--preload` trades RAM for zero first-challenge
  latency; `--device cpu` keeps a player off the GPU.

## 5. Behavioral contract (the LLM depends on it)

Changing any of these is a user-visible API change; update the JSON, the README
and tell the user:

- usernames are stable and unique; `toID(username)` ≤ 18 chars; `MM-` prefix;
- every player accepts challenges in `gen9ou` from any user;
- `players.json` lists exactly the players that were launched; fields and
  `challenge` string stay as documented in §3.7;
- team set is `competitive/gen9ou` filtered by `LocalServerTeamSet`;
- one battle at a time; a challenge sent while the target is offline is lost.

## 6. Engine-level workarounds (inside `rl_player`, not patches)

| symptom | cause | fix |
|---|---|---|
| `Player.battle_against(...)` hangs with 0 CPU | nested `handle_threaded_coroutines` scheduling on `POKE_LOOP` in this poke-env build | use explicit `send_challenges` + `accept_challenges` (as `smoke_test.py`/`heuristic_server.py` do) |
| accept loop never accepts | `accept_challenges` called from inside `POKE_LOOP` nests two scheduling hops | run it from its own thread with `asyncio.run(...)` |
| player gives up after ~5 s | `OpenAIGymEnv._INIT_RETRIES = 250` | `AcceptAnyChallenge._INIT_RETRIES = 10**9` |
| lazy-load hook never fires | `PSClient` captures `player._handle_challenge_request` at init | patch both attributes |
| battle ends but `interact` never returns | `DummyAsyncVectorEnv.step` auto-resets into the next battle | `SingleBattleVectorEnv` |
| `NameError: PokeAgentPlayer is not defined` | upstream `metamon/env/wrappers.py` missing import | injected in `rl_player.py`; fix upstream only if the user agrees |
| `metamon.rl.evaluate` crashes on this box | it never applies `compat.py` and uses `battle_against` | use `player_pool` entry points, or apply compat before calling it |

## 7. Verification matrix

| what changed | run |
|---|---|
| fresh checkout / first run | follow [Before you start](#before-you-start-clone-and-run-the-showdown-server), then `smoke_test --models Minikazam` |
| `compat.py` | all six `check_compat` cases; `check_forward` for Kakuna (flash-attn) **and** Minikazam (GRU) |
| `rl_player.py` / AMAGO loop | `check_forward --agent <model>`; `smoke_test --models <model>` (real battle) |
| `registry.py` / usernames | `python -m player_pool.registry`; `pool --dry-run`; `smoke_test` for the new model |
| `teams.py` | `smoke_test` (the battle must actually start; a rejected team shows as no battle) |
| `pool.py` | `--dry-run`; short `--duration` run; kill one player and confirm restart in `status.json` |
| docs only | no run needed |

Reference costs on this machine: one `check_forward` 5–30 s, one battle
≈ 50–60 s, all 26 players online ≈ 10 s.

## 8. Rules for future changes

Ask the user before:

- downloading more than ~1 GB, or deleting/rebuilding `metamon_cache/`;
- restarting or reconfiguring the shared server instance the user is running
  (starting your own clone per [Before you start](#before-you-start-clone-and-run-the-showdown-server) is expected);
- changing usernames, the format, or pool membership that the LLM sees
  (`players.json`);
- editing tracked upstream files (`metamon/**`, `pyproject.toml`) or adding
  dependencies, when the change can live in `player_pool/`;
- running all 10 policies on a GPU with limited VRAM.

Do not:

- revert the local edits in §3.10 or the patches in §3.6 — each one fixes a
  reproducible failure;
- commit `.venv/`, `metamon_cache/`, `player_pool/logs/`,
  `player_pool/status.json`, `uv.lock` or `*.ipynb`;
- hardcode `cuda`; keep CPU working and use `--device`;
- leave players/pools running when done;
- create temp directories in the repo root on this machine (0o700 DACLs make
  them undeletable);
- pipe native command output in this sandbox (`python ... | Select-String` is
  denied); redirect to a file instead.

Do:

- start your own Showdown clone before any server-dependent test (see
  [Before you start](#before-you-start-clone-and-run-the-showdown-server)); the
  pool and test scripts never launch it;
- keep new behavior in `player_pool/` and treat upstream as read-only;
- use `print(..., flush=True)` for long-running player logs;
- keep `registry.py`, `players.json` and the README in sync;
- keep compat patches conditional so CUDA machines keep native flash-attn and
  `torch.compile`;
- run the ladder in §7 for the layer you changed, and a real battle whenever the
  server is reachable.

## 9. Known gaps / open decisions

- **Format**: only `gen9ou` exists on the stock pinned server
  (`gen9oulongtimer` is not defined there). If the LLM agent needs another
  format, add it in the cloned server's `config/custom-formats.ts`, restart the
  server, and pass `--format <id>` to the pool.
- **Pool size**: 26 = the Gen9OU-capable subset. The paper's "40+ baselines"
  includes Gen1–4-only checkpoints; adding them as Gen9 opponents is possible
  but they would be out of distribution.
- **Ghost usernames**: killing a process can leave the name reserved briefly;
  the supervisor backoff handles production, manual tests should use fresh
  names.
- **GPU memory**: each RL player is a separate process with its own CUDA
  context; ten loaded policies can need 15–20 GB. Use `--only-rl` subsets,
  `--preload` selectively, or `--device cpu`.
- **CPU attention fallback** is float32 with the configured window; it matches
  the native kernel only up to numerics, and long sequences (> window) are the
  approximation. It is not used when native flash-attn is available.

## 10. Where to read more

- [README.md](README.md) — upstream docs: install, datasets, team sets,
  observation/action spaces, training, evaluation.
- [player_pool/README.md](player_pool/README.md) — pool usage, architecture,
  CPU/CUDA behavior, tests.
- [metamon/rl/evaluate/README.md](metamon/rl/evaluate/README.md) — upstream
  evaluation modes and YAML launchers.
- `metamon/rl/pretrained.py` — model registry and per-model docstrings.
