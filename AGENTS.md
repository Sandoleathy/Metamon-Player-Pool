# pi_agent — tool-using LLM player for the local Gen 9 OU pool

This checkout combines:
- the upstream metamon baseline pool (`player_pool/`), and
- a Pi **tool-using agent** (`pi_agent/`) that uses LLMs on the same local Showdown
  battles.

`project.tex` describes the research framing in full. Do not change the upstream
dataset / training code (`metamon/**`) inside this project.

## Project map

```
pi_agent/
  config.py     # constants and environment configuration
  provider.py   # OpenCode Go model provider
  prompt.py     # battle state and action list for the model
  tools.py      # Pi agent tools (state / actions / damage / knowledge)
  player.py     # poke-env Player with a Pi agent chosen move
  run.py        # CLI to battle a registered pool opponent
  knowledge.py  # pi-knowledge tool client
  knowledge_bridge.mjs # Node tool to use local pi-knowledge index
  index_kb.py   # knowledge base index refresh CLI
  build_corpus.py # deterministic Gen 9 OU card generator
  kb/gen9ou/    # generated knowledge cards (not tracked)
  check.py      # offline check of the Pi agent runtime
```

## Runtime contract

- Battle format: `gen9ou` (pinned in `player_pool/registry.py`).
- Model provider: OpenCode Go is the default adapter (`opencode.ai/inference/go`).
  API key must be set via `OPENCODE_API_KEY` / `OPENCODE_GO_API_KEY`.
- Player: `pi_agent.player.create_player` returns a poke-env Player that calls the
  Pi agent one turn at a time.
- The Pi-loop tools are:
  - `get_battle_state`
  - `get_available_actions`
  - `calculate_damage`
  - `search_knowledge` (optional, `pi-knowledge`-backed).
- A failed model call falls back to the highest-power legal move. A failed
  knowledge call is reported as `"KNOWLEDGE UNAVAILABLE:..."` to the model, not a crash.

## Config defaults

| env var | purpose |
|---|---|
| `OPENCODE_API_KEY` | OpenCode Go API key, fallback `OPENCODE_GO_API_KEY` |
| `OPENCODE_MODEL` | model id, default `deepseek-v4.1-flash` |
| `OPENCODE_BASE_URL` | override the endpoint root, default `https://opencode.ai/inference/go/openai/v1` |
| `PI_AGENT_MODEL` | optional model id override (default from `OPENCODE_MODEL`) |
| `PI_AGENT_RAG` | enable the knowledge tool (`1`/`0`) |
| `PI_AGENT_THINKING` | Pi thinking level (`off`/`medium`/...) |
| `PI_KNOWLEDGE_NAME` | knowledge base name, default `gen9ou` |
| `PI_KNOWLEDGE_ROOT` | path to the installed `pi-knowledge` package |
| `PI_KNOWLEDGE_DIR` | pi-knowledge storage directory (default `~/.pi/knowledge`) |
| `PI_AGENT_TRACE_DIR` | decision trace directory, default `pi_agent/logs` |

## Quick start

1. `pip install -e .[agent]` in your project environment; `pi-knowledge` is optional
   and must be installed separately:
   ```bash
   # optional, inside pi_agent/:
   npm install pi-knowledge
   export PI_KNOWLEDGE_ROOT=$PWD/pi_agent/node_modules/pi-knowledge
   ```

2. Build the deterministic knowledge cards and index them:
   ```bash
   python -m pi_agent.build_corpus --out pi_agent/kb/gen9ou
   python -m pi_agent.index_kb --corpus pi_agent/kb/gen9ou
   python -m pi_agent.knowledge_doctor --root $PI_KNOWLEDGE_ROOT
   ```

3. Run one game against a pool opponent (needs the Showdown server):
   ```bash
   python -m pi_agent.run --opponent MM-Kakuna --rag on
   python -m pi_agent.run --opponent MM-Kakuna --rag off
   ```

4. Offline checks (no Showdown server):
   ```bash
   python -m pi_agent.check
   python -m player_pool.check_llm_client
   ```

## Ops

- Run one battle at a time against a registered opponent; do not use the public
  PokéAgent server.
- The pool CLI remains the authoritative way to start opponent processes:
  - `python -m player_pool.pool`
  - `python -m player_pool.pool --dry-run`
- `player_pool/status.json` is transient.

## Coding rules for agents

1. **Keep the changes local to this checkout.** Use `player_pool/` and
   `pi_agent/` instead of editing upstream `metamon/**` code.
2. **Prefer deterministic over interactive.** Do not add hidden controls;
   explicit flags and environment variables are the only paths to external state.
3. **Keep the Pi tool list small.** The tool set exists to solve state parsing
   damage/ability lookups; do this in code rather than asking the model.
4. **Failures are data, not exceptions.** Tools always return a string; only
   the cosmetic `KNOWLEDGE UNAVAILABLE` message changes what the model sees.
5. **No hard-coded dataset paths.** Use the `metamon` package paths for static
   data and the `pi_agent/kb/` tree for generated cards.
6. **No network calls in offline checks.** `pi_agent/check.py` never needs
   Showdown, OpenCode, or pi-knowledge.
7. **Don't regress the OpenCode Go contract.** The provider is OpenAI-compatible
   and must keep the `x-opencode-session` + `User-Agent` behavior present in
   `player_pool/llm_client.py`.

## Project rules of conduct

- Integrations in `pi_agent/` are **tool-level** and **turn-level**: never
  hardcode a specific model identity; configure via the environment.
- Do not commit generated artifacts (`pi_agent/logs/`, `pi_agent/kb/`,
  `.env`).
- Respect the project contract: battle simulators (`metamon`-style) remain the
  source of truth; the LLM is a decision layer, not an evaluation oracle.
