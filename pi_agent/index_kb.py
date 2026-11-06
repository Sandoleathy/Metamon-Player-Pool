"""Index a generated pi_agent corpus into the local ``pi-knowledge`` store.

    python -m pi_agent.index_kb
    python -m pi_agent.index_kb --corpus pi_agent/kb/gen9ou --name gen9ou
    python -m pi_agent.index_kb --doctor --name gen9ou

pi-knowledge silently falls back to BM25 when the stored embedding signature no
longer matches the configured model; run this script after each embedding change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .knowledge import KnowledgeClient, KnowledgeError

DEFAULT_CORPUS = Path(__file__).resolve().parent / "kb" / "gen9ou"


def _client(args: argparse.Namespace) -> KnowledgeClient:
    if args.root:
        os.environ["PI_KNOWLEDGE_ROOT"] = args.root
    if args.module:
        os.environ["PI_KNOWLEDGE_MODULE"] = args.module
    if args.knowledge_dir:
        os.environ["PI_KNOWLEDGE_DIR"] = args.knowledge_dir
    return KnowledgeClient(
        name=args.name,
        knowledge_dir=args.knowledge_dir,
        knowledge_root=args.root,
        module=args.module,
        timeout=args.timeout,
    )


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--name", default=os.environ.get("PI_KNOWLEDGE_NAME", "gen9ou"))
    parser.add_argument("--knowledge-dir", default=None)
    parser.add_argument("--module", default=None, help="module id to import pi-knowledge from")
    parser.add_argument("--root", default=None, help="path to the pi-knowledge package root")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args(argv)
    corpus = Path(args.corpus)
    client = _client(args)
    if not client.available:
        raise SystemExit(
            "pi-knowledge bridge unavailable: install the Node package inside "
            "pi_agent/ (`npm install pi-knowledge`) or set PI_KNOWLEDGE_ROOT."
        )
    if args.doctor:
        try:
            _print(client.doctor())
            return 0
        except KnowledgeError as exc:
            print(f"doctor failed: {exc}")
            return 1

    existing = {kb.get("name") or kb.get("id") for kb in client.list()}
    if args.name in existing:
        if args.force_reindex:
            target = next(kb["id"] for kb in client.list() if kb.get("name") == args.name)
            try:
                client._call(
                    "remove",
                    {"target": target, "remove_vector_file": True},
                )
            except KnowledgeError as exc:
                print(f"remove failed: {exc}")
                return 1
        else:
            try:
                result = client.update(args.name)
                _print(result)
                return 0
            except KnowledgeError as exc:
                print(f"knowledge_update failed: {exc}")
                return 1

    try:
        result = client.add(corpus)
    except KnowledgeError as exc:
        print(f"knowledge_add failed: {exc}")
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
