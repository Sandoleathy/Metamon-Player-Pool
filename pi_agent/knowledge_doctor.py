"""CLI wrapper for ``pi-knowledge``'s ``knowledge_doctor`` health report."""
from __future__ import annotations

import argparse
import json
import os

from .knowledge import KnowledgeClient, KnowledgeError


def _client(args: argparse.Namespace):
    if args.root:
        os.environ["PI_KNOWLEDGE_ROOT"] = args.root
    return KnowledgeClient(
        name=args.name,
        knowledge_dir=args.knowledge_dir,
        knowledge_root=args.root,
        module=args.module,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=os.environ.get("PI_KNOWLEDGE_NAME", "gen9ou"))
    parser.add_argument("--knowledge-dir", default=None)
    parser.add_argument("--module", default=None, help="module id to import pi-knowledge from")
    parser.add_argument("--root", default=None, help="path to the pi-knowledge package root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    client = _client(args)
    try:
        report = client.doctor()
    except KnowledgeError as exc:
        print(f"doctor failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(report.get("report") or report, indent=2))
    else:
        data = report.get("report") or report
        print(f"health score: {data.get('health_score')}/100")
        for issue in data.get("issues", []):
            severity = issue.get("severity", "?")

            kb = issue.get("kb_name") and f" [{issue['kb_name']}]"
            print(f"{severity}{kb or ''}: {issue.get('message')} -> {issue.get('action')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
