# Local Gen9 OU opponent pool

The opponent side of the **AI-Trustworthy LLM Pokémon agent** project
(`project.tex`, CISC7404, University of Macau): a local ladder of scripted
opponents for the E1 experiments ("agent vs. local metamon pool") and for
development. It is a self-contained pool for the local Pokémon Showdown
server, reproducing the *Battling Track* opponent pool of the PokéAgent
Challenge paper. The public challenge server is closed; this local pool is the
replacement:

* **10 metamon RL policies** trained for Gen 9 OU (`Abra`, `Kadabra*`,
  `Alakazam`, `Minikazam`, `Superkazam`, `Kakuna`, `SmallRLGen9Beta`).
* **16 heuristic / IL baselines** registered in `metamon.baselines`
  (`RandomBaseline`, `Grunt`, `GymLeader`, `EmeraldKaizo`, `PokeEnvHeuristic`,
  the three RNN policies, …).

These 26 accounts are the Gen9OU-capable subset of the paper's 40+ baselines;
Gen1–4-only checkpoints stay out of distribution on this server.

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

2. **Local Python environment** with the project installed into it (e.g. virtual
   env or conda; `pip install -e .[agent]` adds the optional pi-python-core
   runtime for the LLM player). Checkpoints and team sets are downloaded from
   HF into `metamon_cache/` (see [AGENTS.md](../AGENTS.md) for naming).

   ```shell
   pip install -e .[agent]   # adds pi-python-core; pi-knowledge is optional
   ```

3. **Checkpoints** (~2.8 GB, all 10 done on this machine, 2026-11-05):

   ```shell
   python -m player_pool.download_models   # 10/10 ready
   ```

   They are stored in `metamon_cache/pretrained_models` (the standard
   HuggingFace cache used by `metamon.rl.pretrained`). On this machine the
   directory is a symlink into the big data volume
   (`~/data/metamon-player-pool/metamon_cache`) so the root disk, which is
   nearly full, is not touched by multi-GB data.

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

```shell
# no server and no API key needed: mock-server checks for the LLM client
python -m player_pool.check_llm_client

# live, keyless: fetches the OpenCode Go model catalog
python -m player_pool.llm_client --list-models
```

---

## LLM access for the agent (OpenCode Go)

The agent's model backend (the report's DeepSeek-direct calls) can be routed
through the OpenCode Console instead of `api.deepseek.com`. OpenCode serves a
single OpenAI-compatible surface for every Go text model, including the same
DeepSeek family (`deepseek-v4.1-flash`, `deepseek-v4-pro`, off-peak/peak). This
matters for the evaluation runs: the same prompts can run direct (pay-as-go
balance) or under a Go/Go Plus/Go Teams monthly allowance, and E1–E4 cells are
long enough that either can be the cheaper option.

[`llm_client.py`](llm_client.py) implements this endpoint with the standard
library only (no new dependency):

| | |
|---|---|
| providers | `opencode-go` (default, monthly allowance) · `opencode` (pay-as-you-go balance) |
| base URL | `https://opencode.ai/inference/go` |
| endpoint | `POST /openai/v1/chat/completions` (Console translates Anthropic/Gemini models) |
| auth | `Authorization: Bearer $OPENCODE_API_KEY` (`oc_sk_…` from the OpenCode Console; `OPENCODE_GO_API_KEY` also accepted) |
| session | `x-opencode-session: ses_…` — **required by Go** (missing it is a `400 MissingSessionID`), generated per client |
| identity | `User-Agent: metamon-player-pool/0.1.0` — OpenCode asks non-OpenCode clients to identify themselves |

```shell
export OPENCODE_API_KEY=oc_sk_...
export OPENCODE_MODEL=deepseek-v4.1-flash

python -m player_pool.llm_client --prompt "Pick the best lead." --json
python -m player_pool.llm_client --model glm-5.3 --list-models
```

```python
from player_pool.llm_client import LLMClient

client = LLMClient(model="deepseek-v4.1-flash")  # reads OPENCODE_API_KEY
reply = client.chat(
    [{"role": "user", "content": "Choose a move."}],
    system="You are a Gen9OU player.",
    temperature=0.3,
    max_tokens=256,
)
print(reply.text)
print(reply.usage.allowance)   # dollars of the Go monthly limit this request used

reply = await client.achat(...)  # same call from the agent's event loop
```

Agent-relevant details:

* **Key storage**: `OPENCODE_API_KEY` from the shell, or the repo-root `.env`
  (gitignored). The client loads `.env` itself — no dotenv import needed.
  A working Go key is committed locally as of 2026-11-05 and verified with a
  real request (`billing_source=go`, `cost=0` inside the plan).
* One `LLMClient` = one conversation. OpenCode uses the session id for routing
  and prompt caching, and Go *requires* it. For a battle agent that means one
  client per game; pass `session_id=` explicitly if the conversation outlives
  a process.
* `usage.opencode` in every response reports `cost`, `billing_source`
  (`go`, `go-plus`, `go-teams`, `balance`, `free`, `byok`) and `allowance`, so
  the decision trace can log per-turn spend exactly like the report's trace
  format.
* Free-tier chat models (`beam-free`, `*-free` ids) answer `HTTP 403` outside
  OpenCode itself, even with a key — use paid ids from this client.
* Retries: 408/409/425/429/5xx and network errors retry up to
  `max_retries` (honouring `Retry-After`); 4xx errors raise `LLMError(status,
  error_type)` immediately. No key raises `MissingAPIKeyError` before any
  network call.
* Need your own provider account instead of Go? Use
  `LLMClient(provider=..., base_url=...)` only for OpenCode-compatible roots
  (`https://opencode.ai/inference`). Direct `api.deepseek.com` calls keep using
  the harness's own DeepSeek client for now — see
  [Notes and limitations](#notes-and-limitations).

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
├── llm_client.py        OpenCode Go/Console chat client for the agent (stdlib only)
├── check_llm_client.py  offline checks for llm_client (mock server)
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

## Running on any machine

The pool code is platform- and device-agnostic; only the environment setup
changes on a *new* machine. On a CUDA box it uses flash-attn 2 when it is
available; on a CPU-only box it uses the CPU fallbacks in `compat.py`.

1. Create the environment and install:

   ```shell
   python -m venv .venv && . .venv/bin/activate
   pip install -e .
   # optionally add flash-attn 2 on a CUDA box:
   pip install flash-attn --no-build-isolation
   ```

2. The AMAGO experiment automatically places the policy on `cuda` when
   `torch.cuda.is_available()` (via `accelerate`), so no code change is needed.

3. The compatibility patches adapt automatically:

   Override with `PLAYER_POOL_FLASH_ATTENTION=native|cpu` and
   `PLAYER_POOL_TORCH_COMPILE=0|1` if needed. `check_compat` covers the cases
   above.

4. **GPU memory**: each RL player is a separate process, so each loaded policy
   costs its weights plus one CUDA context (~0.3–0.6 GB on top of the model).
   Ten loaded policies can need 15–20 GB; lazy loading means only challenged
   models are resident. On a smaller GPU use `--only-rl` to run a subset, or
   `--device cpu` to keep the pool on CPU and run selected models on the GPU in
   separate `rl_player` processes.

---

## Notes and limitations

* **CPU/GPU plug-ins.** Environment-specific patches are described above; no
  memory tuning is needed unless a very limited machine is involved.
* **Sampling.** Policies act with temperature 1.0 (`--temperature` to change),
  matching normal metamon evaluation (`sample=True`).
* **Ghost users.** If a player process is killed, Showdown can keep the name
  reserved for a short while (`|nametaken|Someone is already using the name`).
  The supervisor's restart backoff handles this; when testing manually, use a
  fresh username or wait for the old connection to time out.
* **Server restarts.** All players reconnect automatically: the supervisor
  restarts RL processes; heuristic players recreate themselves inside the
  process.
