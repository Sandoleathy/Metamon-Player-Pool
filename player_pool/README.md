# Local Gen9 OU opponent pool

A self-contained player pool for the local Pokémon Showdown server, built to
reproduce the *Battling Track* opponent pool of the PokéAgent Challenge paper:

* **10 metamon RL policies** trained for Gen 9 OU (`Abra`, `Kadabra*`,
  `Alakazam`, `Minikazam`, `Superkazam`, `Kakuna`, `SmallRLGen9Beta`).
* **16 heuristic / IL baselines** registered in `metamon.baselines`
  (`RandomBaseline`, `Grunt`, `GymLeader`, `EmeraldKaizo`, `PokeEnvHeuristic`,
  the three RNN policies, …).

Every player is an ordinary Showdown account that idles on
`ws://localhost:8000/showdown/websocket`, accepts challenges from anyone in
`gen9ou`, and keeps playing until stopped. An LLM agent picks an opponent from
[`players.json`](players.json) and challenges it by username.

```
python -m player_pool.pool                 # start the whole pool
python -m player_pool.pool --dry-run       # show what would be launched
```

`players.json` is written on startup and contains, for each player:

```json
{
  "username": "MM-Kakuna",
  "format": "gen9ou",
  "kind": "rl",
  "agent": "Kakuna",
  "checkpoint": 34,
  "params": "142M",
  "strength": "Gen9OU GXE ~71% vs humans with sample teams (paper Table 1).",
  "challenge": "/challenge MM-Kakuna, gen9ou"
}
```

The agent can therefore pick uniformly at random, by strength, or by family.
`status.json` shows which pool processes are currently alive.

---

## Requirements

1. **Local Showdown server on port 8000 started with `--no-security`.**

   The pool does not ship or launch a server: clone the pinned server yourself
   (see [AGENTS.md](../AGENTS.md#before-you-start-clone-and-run-the-showdown-server)),
   then start it:

   ```shell
   # one-time: git submodule update --init server/pokemon-showdown
   #           (or clone smogon/pokemon-showdown at the pinned commit)
   cd server/pokemon-showdown
   npm install
   node pokemon-showdown start --no-security
   ```

   Without `--no-security` the server rejects guest logins with
   `|nametaken|...|Your authentication token was invalid.` and no player can
   come online.

2. **Python environment** with `metamon`, `amago`, `poke-env`, `torch`
   (the repository's `.venv` already has them).

3. **Checkpoints** (~2.8 GB for one checkpoint of each model):

   ```shell
   python -m player_pool.download_models
   ```

   They are stored in `metamon_cache/pretrained_models` (the standard
   HuggingFace cache used by `metamon.rl.pretrained`).

4. **Team set**: `competitive/gen9ou` is downloaded automatically on first use
   (`get_metamon_teams`). Teams that the local ruleset forbids (Tera Blast,
   Zoroark/Zorua, Revival Blessing) are filtered out by
   [`teams.py`](teams.py).

---

## Usage

```shell
# everything: 1 heuristic process + 1 process per RL policy
python -m player_pool.pool

# only some players
python -m player_pool.pool --only-rl Kakuna Abra --only-heuristic Grunt GymLeader

# only RL / only heuristics
python -m player_pool.pool --rl-only
python -m player_pool.pool --no-heuristics

# load selected checkpoints at startup instead of on first challenge
python -m player_pool.pool --preload Kakuna Kadabra3 Abra

# keep it running for an hour and then stop
python -m player_pool.pool --duration 3600

# force the RL policies onto CPU even on a CUDA machine
python -m player_pool.pool --device cpu
# one-off player with an explicit device
python -m player_pool.rl_player --agent Kakuna --device cpu
```

Logs go to `player_pool/logs/` (one file per player). The supervisor restarts
crashed players with exponential backoff and rewrites `status.json` every 5 s.

### Single players

```shell
# one RL policy (lazy load)
python -m player_pool.rl_player --agent Kakuna --username MM-Kakuna

# one RL policy, preloaded, stop after one battle
python -m player_pool.rl_player --agent Abra --preload --max-battles 1

# all heuristic baselines in one process
python -m player_pool.heuristic_server --format gen9ou
```

### Tests

```shell
# no server needed: load a checkpoint and run a few policy steps on synthetic obs
# (exercises the tstep/traj encoders, FlashAttention fallback and KV cache)
python -m player_pool.check_forward --agent Kakuna

# no server needed: verify patch selection for CPU / CUDA+flash-attn / forced modes
python -m player_pool.check_compat auto_cuda_with_flash

# needs the Showdown server: play one battle against every RL model
python -m player_pool.smoke_test
```

`check_forward` is the recommended first check after moving the project to a
CUDA machine, before starting the pool.

---

## How it works

```
player_pool/
├── registry.py          player specs (username, checkpoint, description) + players.json
├── teams.py             TeamSet filtered for the local gen9ou bans
├── rl_player.py         one AMAGO checkpoint as an always-on Player (lazy load)
├── heuristic_server.py  all heuristic baselines in one asyncio process
├── pool.py              supervisor: starts/restarts processes, writes player/status json
├── download_models.py   prefetch checkpoints from HuggingFace
├── smoke_test.py        end-to-end battle test per RL model
├── compat.py            local workarounds (see below)
├── players.json         generated registry for the LLM agent
└── status.json          generated process status
```

**RL players.** `rl_player.py` wraps a metamon policy in the same gym/AMAGO
stack that `metamon.rl.evaluate` uses, but keeps the player connected
indefinitely and only accepts challenges. The checkpoint is loaded lazily on
the first challenge request: the hook on `_handle_challenge_request` loads the
AMAGO experiment before the challenge is accepted, so the model is not loaded
for players that nobody challenges and no battle timer is running during the
load.

After each battle the player immediately waits for the next challenge on the
same connection.

**Heuristic players.** One process hosts all 16 baselines; each is a normal
`poke_env.Player` running `accept_challenges(None, ...)`. A dropped baseline is
recreated automatically.

**Both formats.** The server accepts challenges in `gen9ou` only (probed with
`/challenge`; `gen9oulongtimer` and similar ids are *not* registered on this
server, although `/search` reports them as "not ladderable"). If the LLM agent
uses a different format id, pass `--format <id>` to the pool and the AI must use
the same id. Note that metamon aliases `gen9oulongtimer` to `gen9ou` for
observation purposes, but poke-env's challenge filter matches the exact format
string.

---

## Local workarounds in `compat.py`

These work around environment issues, not metamon bugs:

| Patch | Why |
|---|---|
| `patch_mkdir_permissions` | Python 3.14 on Windows turns `os.mkdir(..., 0o700)` into a restrictive DACL, so `tempfile.mkdtemp` directories become unwritable and every `huggingface_hub` download fails. Forces `0o777`. No-op off Windows. |
| `patch_torch_dynamo` | amago decorates networks with `torch.compile`; Inductor needs a C++ toolchain and piped subprocesses. Disabled by default on Windows and on CPU-only machines; **left enabled on a CUDA box**. Force with `PLAYER_POOL_TORCH_COMPILE=0/1`. |
| `patch_flash_attention` | 8 of 10 RL policies use `amago.nets.transformer.FlashAttention` (flash-attn 2 is CUDA-only). The weights are shared with `SelfAttention`, so the forward pass is replaced by a plain PyTorch implementation that keeps the configured causal + sliding-window mask and the KV cache. **Only installed when native flash-attn cannot run** (not installed, or no CUDA device); a CUDA box with flash-attn 2 keeps the upstream fast path. Force with `PLAYER_POOL_FLASH_ATTENTION=native/cpu`. |
| `patch_amago_traj_length` | amago's default `traj_save_len = 1e10` is a float, which `random.randint` rejects during `Experiment.start()`. Needed everywhere. |

`rl_player.py` also fills in `metamon.env.wrappers.PokeAgentPlayer`, which the
upstream module references but forgets to import.

---

## Running on a CUDA machine

The pool code is platform- and device-agnostic; only the environment setup changes:

1. Create a fresh environment on the new machine and install the repo with the
   CUDA build of PyTorch (the checked-in `.venv` is a Windows CPU build and is
   not portable):

   ```shell
   python -m venv .venv && . .venv/bin/activate
   pip install -e .
   # install a CUDA torch matching the driver, then optionally flash-attn 2:
   pip install flash-attn --no-build-isolation   # optional, big speedup
   ```

2. The AMAGO experiment automatically places the policy on `cuda` when
   `torch.cuda.is_available()` (via `accelerate`), so no code change is needed.

3. The compatibility patches adapt automatically:

   | | CPU box (this machine) | CUDA box + flash-attn | CUDA box, no flash-attn |
   |---|---|---|---|
   | attention | CPU fallback | native flash-attn | CPU fallback (runs on GPU) |
   | `torch.compile` | disabled | enabled | enabled |

   Override with `PLAYER_POOL_FLASH_ATTENTION=native|cpu` and
   `PLAYER_POOL_TORCH_COMPILE=0|1` if needed.

4. **GPU memory**: each RL player is a separate process, so each loaded policy
   costs its weights plus one CUDA context (~0.3–0.6 GB on top of the model).
   Ten loaded policies can need 15–20 GB; lazy loading means only challenged
   models are resident. On a smaller GPU use `--only-rl` to run a subset, or
   `--device cpu` to keep the pool on CPU and run selected models on the GPU in
   separate `rl_player` processes.

5. On Linux the supervisor uses `SIGTERM` and does not pass
   `CREATE_NEW_PROCESS_GROUP`; everything else (ports, usernames, `players.json`)
   is identical. If the checkpoints were not copied with the project, rerun
   `python -m player_pool.download_models`.

---

## Notes and limitations

* **CPU by default here.** The local venv has a CPU build of torch. Kakuna/Superkazam
  (142M) take ~4–6 s to load and roughly 0.1–0.3 s per move; the 57M models are
  faster. Ten RL processes idle at a few hundred MB each; models are loaded
  only when challenged. On a CUDA box the same players run on the GPU (see the
  section above).
* **FlashAttention approximation (CPU / no-flash-attn only).** The fallback
  sliding-window attention is computed in float32 with the same window size;
  outputs match the GPU path up to numerics, but long sequences (> window) are
  the interesting case. With flash-attn 2 installed on CUDA, the native kernel
  is used instead.
* **Sampling.** Policies act with temperature 1.0 (`--temperature` to change),
  matching normal metamon evaluation (`sample=True`).
* **Ghost users.** If a player process is killed, Showdown can keep the name
  reserved for a short while (`|nametaken|Someone is already using the name`).
  The supervisor's restart backoff handles this; when testing manually, use a
  fresh username or wait for the old connection to time out.
* **Server restarts.** All players reconnect automatically: the supervisor
  restarts RL processes; heuristic players recreate themselves inside the
  process.
