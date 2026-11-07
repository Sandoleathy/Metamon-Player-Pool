# `pi_agent` — tool-using LLM player for the local Gen 9 OU pool

`pi_agent` runs one Pokemon Showdown battle turn at a time through a
`pi-python-core` agent. The model gets the battle state, a list of legal
actions, and up to four tools; it answers with one JSON action that is mapped
back to a poke-env `BattleOrder`.

The agent talks to the local Showdown server and challenges the players of
[`player_pool/`](../player_pool/README.md). Nothing here requires the public
PokéAgent server.

## Runtime layout

| File | Role |
| --- | --- |
| `run.py` | CLI entry point: connect, challenge one pool opponent, report W/L |
| `player.py` | poke-env `Player` subclass; the per-turn decision loop and fallbacks |
| `prompt.py` | state prompt, action list, tolerant answer parsing, repair prompt |
| `tools.py` | the four model-callable tools |
| `damage.py` | damage / type / KO / turn-order evidence for `calculate_damage` |
| `provider.py` | OpenCode Go (OpenAI-compatible) provider and agent construction |
| `config.py` | environment-driven `AgentConfig` with clamped limits |
| `trace.py` | one JSONL record per decided turn under `pi_agent/logs/` |
| `knowledge.py` | client for a local `pi-knowledge` index |
| `knowledge_bridge.mjs` | Node bridge that imports the `pi-knowledge` engine |
| `build_corpus.py` | deterministic Gen 9 OU knowledge cards |
| `index_kb.py` | build / update / doctor the `pi-knowledge` index |
| `knowledge_doctor.py` | health report for the index |
| `check.py` | offline checks: no server, no API key |

## Requirements

- Python 3.10+ with the project installed, including the agent extra:

  ```bash
  pip install -e .[agent]        # from the repository root
  ```

- A local Pokemon Showdown server (the copy in `server/pokemon-showdown`).
- Node.js 22+ only if you want the `search_knowledge` tool.
- An API key for the model endpoint (`OPENCODE_API_KEY`).

## Quick start

1. Start the Showdown server:

   ```bash
   cd server/pokemon-showdown
   npm install
   node pokemon-showdown start --no-security
   ```

2. Start the opponent pool from the repository root:

   ```bash
   python -m player_pool.pool --dry-run   # check the pool config first
   python -m player_pool.pool
   ```

3. Optional — build and index the knowledge cards (see below).

4. Play one game against a pool opponent:

   ```bash
   export OPENCODE_API_KEY=oc_sk_...
   python -m pi_agent.run --opponent MM-Kakuna --rag off
   python -m pi_agent.run --opponent MM-Kakuna --rag on
   ```

   Useful flags: `--username` (Showdown account, max 18 characters),
   `--model`, `--thinking off|medium|high`, `--trace-dir`.

5. Offline checks (no server, no key, no Node):

   ```bash
   python -m pi_agent.check
   ```

## How one turn is decided

1. If there is exactly one legal order (a single move, or a single forced
   switch), it is returned immediately and no model call is made. The trace
   records `"chosen_kind": "forced"`.
2. Otherwise the state prompt plus the legal-action JSON is sent to a fresh
   agent built with that turn's tools. The agent may call tools until it
   answers.
3. The answer is parsed tolerantly (prose and code fences are tolerated) and
   mapped to a legal order. If it is not legal, one repair prompt is sent that
   names the legal move and switch ids. An empty answer (bad key, dead
   endpoint) skips the repair attempt and goes straight to the fallback.
4. If the second attempt is still not legal, or the model call raises, the
   player falls back to the highest-base-power legal move. A failed model call
   never ends the battle.
5. One JSONL record is appended to `pi_agent/logs/`.

## Tools

| Tool | What the model gets |
| --- | --- |
| `get_battle_state` | the same reduced state text as the turn prompt |
| `get_available_actions` | legal moves and switches as JSON (power, type, priority, Tera legality) |
| `calculate_damage` | expected damage %, type multiplier, STAB, KO estimate, turn order |
| `search_knowledge` | up to `top_k` knowledge cards, filtered by `category` |

`search_knowledge` only exists when `--rag on` (or `PI_AGENT_RAG=1`) and the
knowledge client can be constructed. If the index or Node is missing, the tool
returns `KNOWLEDGE UNAVAILABLE: ...` and the battle continues.

## Knowledge base (optional)

`build_corpus.py` writes one Markdown card per species, move, ability, item and
core mechanic into `pi_agent/kb/gen9ou/`, plus a SHA-256 manifest. The output is
deterministic: no model is involved.

```bash
# 1. install the Node package (once, inside pi_agent/)
npm install pi-knowledge
export PI_KNOWLEDGE_ROOT=$PWD/pi_agent/node_modules/pi-knowledge

# 2. generate the cards
python -m pi_agent.build_corpus --out pi_agent/kb/gen9ou

# 3. index them into pi-knowledge
python -m pi_agent.index_kb --corpus pi_agent/kb/gen9ou

# 4. health check
python -m pi_agent.knowledge_doctor
```

Notes:

- Ability and item cards need Showdown's compiled text data
  (`server/pokemon-showdown/dist`). Without a built server checkout only the
  species, move and mechanic cards are generated.
- Re-run `index_kb` whenever the cards or the embedding model change;
  `pi-knowledge` silently falls back to BM25 when the embedding signature no
  longer matches.
- `pi_agent/kb/` is generated and git-ignored; do not commit it.

## Tracing

One file per game: `pi_agent/logs/<timestamp>-<username>.jsonl`, one line per
decided turn. Fields include `battle`, `turn`, `opponent`, `rag`, `actions`,
`answer`, `parsed`, `chosen`, `chosen_kind` (`model` / `fallback` / `forced`),
`attempts`, `usage`, `thinking`, `stop_reason`, `tools`, `latency_s`.

`tools` is the important one for a RAG comparison: it records which tool ran,
with which arguments, and what it returned.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENCODE_API_KEY` / `OPENCODE_GO_API_KEY` | — | model endpoint key |
| `OPENCODE_BASE_URL` | `https://opencode.ai/inference/go/openai/v1` | endpoint root |
| `OPENCODE_MODEL` / `PI_AGENT_MODEL` | `deepseek-v4.1-flash` | model id |
| `PI_AGENT_THINKING` | `off` | thinking level passed to the agent |
| `PI_AGENT_RAG` | `1` | enable the knowledge tool |
| `PI_AGENT_TRACE_DIR` | `pi_agent/logs` | trace directory |
| `PI_AGENT_MAX_MODEL_REQUESTS` | `8` | per-turn model calls (1–64) |
| `PI_AGENT_MAX_TOOL_CALLS` | `8` | per-turn tool calls (0–64) |
| `PI_AGENT_RUN_TIMEOUT` | `240` | per-turn agent timeout, seconds (5–3600) |
| `PI_AGENT_MAX_TOKENS` | `8192` | output ceiling (1024–32768) |
| `PI_AGENT_CONTEXT_WINDOW` | `128000` | model context window |
| `PI_KNOWLEDGE_NAME` | `gen9ou` | knowledge base name |
| `PI_KNOWLEDGE_ROOT` | — | path to the `pi-knowledge` package |
| `PI_KNOWLEDGE_DIR` | `~/.pi/knowledge` | pi-knowledge storage |

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'pi_python'` | install the extra: `pip install -e .[agent]` |
| `KNOWLEDGE UNAVAILABLE: pi-knowledge bridge unavailable` | no Node package: `npm install pi-knowledge` inside `pi_agent/`, or set `PI_KNOWLEDGE_ROOT` |
| every turn logs `fallback to ...` | the model is unreachable or its answers are not legal; check the trace `attempts` block |
| the challenge never starts | the pool is not running, the opponent name is wrong, or the server rejects the login |
| no trace file appears | `PI_AGENT_TRACE_DIR` is not writable (the battle continues, but nothing is recorded) |

## Limits

- One battle at a time; `max_concurrent_battles=1`.
- The battle timer is disabled while the agent thinks.
- Only the `io` prompting style exists: one JSON action per turn, no search
  tree.
- `pi_agent` never uploads to the public ladder and never starts the Showdown
  server or the pool for you.
