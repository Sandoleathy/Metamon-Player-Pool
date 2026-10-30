"""Team sets for the local server.

The local Showdown ruleset (PokéAgent Challenge rules) rejects a few things that
appear in the cached ``competitive`` team set, e.g. Tera Blast.  Filter those
teams out instead of letting the server reject a battle at team preview.
"""
from __future__ import annotations

from metamon.env import get_metamon_teams
from metamon.env.wrappers import TeamSet

# substrings of the packed team that the local gen9ou ruleset forbids
BANNED_SUBSTRINGS = ("terablast", "zoroark", "zorua", "revivalblessing")


class LocalServerTeamSet(TeamSet):
    """``TeamSet`` that rejects teams the local server would refuse."""

    def block_team(self, packed_team: str) -> bool:
        packed = packed_team.lower()
        return any(token in packed for token in BANNED_SUBSTRINGS)


def get_local_teams(
    battle_format: str = "gen9ou", set_name: str = "competitive"
) -> LocalServerTeamSet:
    return get_metamon_teams(battle_format, set_name, set_type=LocalServerTeamSet)
