"""Deterministic Gen 9 OU knowledge cards for ``pi-knowledge``.

Multi-card knowledge files live in ``pi_agent/kb/gen9ou`` and match the
category names the agent tool understands:

    species/  move/  ability/  item/  mechanic/

The generator is pure Python: metamon ships the species and move tables, and
the optional Showdown server checkout (``server/pokemon-showdown``) supplies
abilities/items/moves English text, exactly as pi-python / pi-knowledge do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "metamon" / "backend" / "showdown_dex" / "static"
SERVER = ROOT / "server" / "pokemon-showdown"
DEFAULT_DATABASE_DIR = ROOT / "pi_agent" / "kb"


def _node_text_export(source_relative: str, body: str) -> Any:
    """Evaluate a CommonJS module in the Showdown server and print JSON."""
    module = (SERVER / source_relative).resolve()
    if not module.is_file():
        return {}
    script = f"const d = require({json.dumps(str(module))}); {body} console.log(JSON.stringify(values));"
    proc = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode:
        return {}
    try:
        return json.loads(proc.stdout.strip())
    except ValueError:
        return {}


def effect_text(kind: str) -> dict[str, str]:
    """Map pokemon id -> `shortDesc` or `desc` text from Showdown's data."""
    compiled = (
        "abilities.js"
        if kind == "abilities"
        else "items.js"
        if kind == "items"
        else "moves.js"
        if kind == "moves"
        else "pokedex.js"
    )
    key = {"abilities": "AbilitiesText", "items": "ItemsText", "moves": "MovesText"}[kind]
    return _node_text_export(
        f"dist/data/text/{compiled}",
        f"const values = {{}}; for (const [id, text] of Object.entries(d.{key})) "
        "values[id] = text.shortDesc || text.desc || '';",
    )


def clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", (value or "").lower()).strip("_")


def species_pokedex() -> dict[str, Any]:
    with (STATIC / "pokemon" / "gen9pokedex.json").open("rb") as file:
        return json.load(file)


def moves_pokedex() -> dict[str, Any]:
    with (STATIC / "moves" / "gen9moves.json").open("rb") as file:
        return json.load(file)


def ability_introductory() -> dict[str, str]:
    """Id key -> effect (best effort: Showdown's texts, else effort-free metadata)."""
    return effect_text("abilities")


def item_effects() -> dict[str, str]:
    return effect_text("items")


def move_effects() -> dict[str, str]:
    return effect_text("moves")


def species_card(name: str, data: dict[str, Any]) -> str:
    stats = data.get("baseStats") or {}
    stats_line = "/".join(f"{x}{stats.get(x, '?')}" for x in ("hp", "atk", "def", "spa", "spd", "spe"))
    abilities = data.get("abilities") or {}
    ability_line = "; ".join(f"{k}:{v}" for k, v in abilities.items())
    types = "/".join(data.get("types") or ["?"])
    return f"# {name}\n\ntype: {types}\nstat: {stats_line}\nability: {ability_line}\n"


def move_card(name: str, data: dict[str, Any], effect: str) -> str:
    flags = "; ".join(sorted(k for k, v in (data.get("flags") or {}).items() if v)) or "-"
    secondary = (data.get("secondary") or {}).get("effect") if isinstance(data.get("secondary"), dict) else None
    effect = effect or secondary or "no effect data"
    base = data.get("basePower") or 0
    accuracy = data.get("accuracy") if isinstance(data.get("accuracy"), (int, float)) else 100
    return (
        f"# {name}\n\n"
        f"type: {data.get('type','?')}\n"
        f"category: {data.get('category','?')}\n"
        f"power: {base}\n"
        f"accuracy: {accuracy}%\n"
        f"priority: {data.get('priority',0)}\n"
        f"target: {data.get('target','?')}\n"
        f"flags: {flags}\n"
        f"effect: {effect}\n"
    )


def ability_card(name: str, effect: str) -> str:
    return f"# {name}\n\neffect: {effect}\n"


def item_card(name: str, effect: str) -> str:
    return f"# {name}\n\neffect: {effect}\n"


def mechanic_cards() -> list[tuple[str, str]]:
    """Hand-written core rule cards."""
    return [
        ("terastallization", "Terastallization sets the pokemon's type to pure Tera type pieces; Gen 9 only. Once per battle."),
        ("terastallize-banned", "Showdown strips Tera types from competitive sample teams when the format eliminates full Terastallization."),
        ("evolution", "Species evolve when they meet `prevo`, `evos`, or `evoItem` requirements. Form changes are not evolution in this data."),
        ("items", "Items affect battle simulation through `item` fields. Held item effects are handled by the server's item data."),
        ("abilities", "Ability effects are applied by the simulator; player-visible data is in `ability_introductions.json`."),
        ("hazards", "Stealth Rock / Spikes / Toxic Spikes / Sticky Web apply before turns. Knock them off by field removal."),
        ("weather-terrain", "Weather and terrain modify move damage / accuracy and lose each other if the opposing type is immune."),
        ("stages", "Boost stages: 0..6 for Attack/Defense/SpA/SpD/Speed/Evasion; negative moves use inverse effective stages."),
        ("switch", "Switching spends a turn and lets hazards apply to the entering Pokemon immediately."),
        ("simulator-turn-order", "Priority -> raw speed after boosts -> within priority usingOrange sort. Forced switch skips speed."),
        ("hazards-order", "Stealth Rock -> Spikes 1-3 -> Toxic Spikes 1-2 -> Sticky Web."),
        ("status", "Burn/Toxic poison/Paralysis/Sleep/Frostbite are different states; each has distinct duration and effects."),
        ("evasion", "Evasion stage affects accuracy via the Showdown formula, not a flat bias."),
        ("tera-type", "Only same-type (single type) tera pieces are relevant in Gen 9 OU; not restricted by types."),
        ("showdown-data-name", "All Showdown IDs and pokemon/move/ability names are lowercase ids; focus search on ids."),
        ("knowledge-grounding", "Battle state text contains only visible information; hidden items/abilities are never named."),
    ]


def write_cards(
    category: str,
    outputs: Iterable[tuple[str, str]],
    out_dir: Path,
    max_chars: int = 1000,
) -> int:
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, body in outputs:
        body = body.strip()
        if not body:
            continue
        if len(body) > max_chars:
            body = body[:max_chars - 1].rstrip() + "\u2026"
        path = category_dir / f"{clean_id(name)}.md"
        path.write_text(body, encoding="utf-8")
        count += 1
    return count


def build(out_dir: Path, max_chars: int = 1000) -> dict[str, int]:
    effects_moves = move_effects()
    effects_abilities = ability_introductory()
    effects_items = item_effects()

    species_out: list[tuple[str, str]] = []
    for sid, data in species_pokedex().items():
        if data.get("isNonstandard") is not None:
            continue
        species_out.append((data.get("name", sid), species_card(data.get("name", sid), data)))

    move_out: list[tuple[str, str]] = []
    for mid, data in moves_pokedex().items():
        if data.get("isNonstandard") is not None:
            continue
        name = data.get("name", mid)
        move_out.append(
            (name, move_card(name, data, effects_moves.get(mid, effects_moves.get(name, ""))))
        )

    ability_out: list[tuple[str, str]] = []
    for name, effect in effects_abilities.items():
        if not effect:
            continue
        ability_out.append((name, ability_card(name, effect)))

    item_out: list[tuple[str, str]] = []
    for name, effect in effects_items.items():
        if not effect:
            continue
        item_out.append((name, item_card(name, effect)))

    mechanics_out: list[tuple[str, str]] = [
        (name, f"# {name.replace('-', ' ').capitalize()}\n\n{effect}") for name, effect in mechanic_cards()
    ]

    counts = {
        "species": write_cards("species", species_out, out_dir, max_chars),
        "move": write_cards("move", move_out, out_dir, max_chars),
        "ability": write_cards("ability", ability_out, out_dir, max_chars),
        "item": write_cards("item", item_out, out_dir, max_chars),
        "mechanic": write_cards("mechanic", mechanics_out, out_dir, max_chars),
    }
    manifest = build_manifest(out_dir)
    (out_dir / "manifest.json").write_text(
        json.dumps({**manifest, "counts": counts, "max_chars": max_chars}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return counts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(out_dir: Path) -> dict[str, Any]:
    """Per-file SHA-256 manifest of the emitted cards."""
    files = sorted(
        path
        for path in out_dir.rglob("*.md")
        if path.is_file()
    )
    return {
        "generated_by": "pi_agent.build_corpus",
        "hashes": {str(p.relative_to(out_dir)): sha256(p) for p in files},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out else DEFAULT_DATABASE_DIR
    counts = build(out_dir)
    print(f"wrote {out_dir}", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
