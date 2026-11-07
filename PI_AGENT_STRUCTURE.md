# pi-agent-battling 结构说明

## ## LLM Battle Agent (pi_agent/)


## LLM Battle Agent (`pi_agent/`)

The `pi_agent` runtime is a small tool-using LLM agent built on `pi-python-core`:

- **Model backend**: OpenCode Go by default (`deepseek-v4.1-flash`),
  with a generic Chat Completions adapter for OpenCode-compatible roots.
- **Battle loop**: reduced Showdown state prompt and poke-env action mapping.
- **Tools**: `get_battle_state`, `get_available_actions`, `calculate_damage`,
  and optional `search_knowledge`.
- **Knowledge**: optional [pi-knowledge](https://pi.dev/packages/pi-knowledge)
  index of 3,030 deterministic cards in `pi_agent/kb/gen9ou/`.
- **Trace**: one JSONL per game under `pi_agent/logs/`.

<br>
