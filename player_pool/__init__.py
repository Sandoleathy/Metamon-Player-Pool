"""Opponent pool for the local Gen9 OU ladder.

Two kinds of players are served on the local Pokémon Showdown server
(``ws://localhost:8000/showdown/websocket``):

* ``rl``        — metamon pretrained policies (AMAGO checkpoints from
                  ``jakegrigsby/metamon`` on HuggingFace), loaded lazily.
* ``heuristic`` — the heuristic / IL baselines registered in
                  ``metamon.baselines``.

The supervisor (``player_pool.pool``) keeps one process per RL policy plus a
single process hosting all heuristic players, and writes ``players.json`` so an
LLM agent can pick an opponent and challenge it by username.

Entry points:
    python -m player_pool.pool            # supervisor (writes players.json)
    python -m player_pool.registry        # registry only
    python -m player_pool.rl_player       # one RL policy
    python -m player_pool.heuristic_server
    python -m player_pool.check_forward   # offline policy forward check
    python -m player_pool.check_compat    # patch-selection self-check
    python -m player_pool.check_llm_client  # offline LLM client checks
    python -m player_pool.llm_client      # OpenCode Go chat client (model backend)
    python -m player_pool.smoke_test      # battle test (needs Showdown server)
"""
