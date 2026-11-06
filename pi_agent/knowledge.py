"""Thin, synchronous client for a local ``pi-knowledge`` index.

The heavy lifting stays in the Node tool: this module shells out to
``knowledge_bridge.mjs``, which imports ``KnowledgeEngine`` from the installed
``pi-knowledge`` package.  Nothing here depends on embedding providers; the
configured ``PI_KNOWLEDGE_EMBEDDING`` is used exactly as pi-knowledge resolves it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BRIDGE = Path(__file__).resolve().parent / "knowledge_bridge.mjs"


class KnowledgeError(RuntimeError):
    """The knowledge tool is unavailable or the index cannot answer."""


@dataclass
class KnowledgeHit:
    content: str
    file_path: str
    file_type: str
    kb_name: str
    score: float
    snippet: str
    start_line: int
    end_line: int
    chunk_id: str | None = None
    match_reason: str | None = None


class KnowledgeClient:
    """Search, (re)index and diagnose a local pi-knowledge knowledge base."""

    def __init__(
        self,
        *,
        name: str = "gen9ou",
        bridge: Path | None = None,
        knowledge_dir: str | None = None,
        knowledge_root: str | None = None,
        module: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.name = name
        self.bridge = bridge or BRIDGE
        self.knowledge_dir = knowledge_dir or os.environ.get("PI_KNOWLEDGE_DIR")
        self.knowledge_root = knowledge_root or os.environ.get("PI_KNOWLEDGE_ROOT")
        self.module = module or os.environ.get("PI_KNOWLEDGE_MODULE") or "pi-knowledge"
        self.timeout = timeout

    # ------------------------------------------------------------------
    # bridge plumbing
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.bridge.is_file() and shutil.which("node") is not None

    def _call(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            raise KnowledgeError(
                "pi-knowledge bridge unavailable. Run "
                "`npm install pi-knowledge` inside pi_agent/ and set "
                "PI_KNOWLEDGE_MODULE if you do not install it in this directory."
            )
        request = dict(payload)
        request["kb_name"] = self.name
        env = dict(os.environ)
        if self.knowledge_dir:
            env["PI_KNOWLEDGE_DIR"] = self.knowledge_dir
        if self.knowledge_root:
            env["PI_KNOWLEDGE_ROOT"] = self.knowledge_root
        if self.module:
            env["PI_KNOWLEDGE_MODULE"] = self.module
        try:
            proc = subprocess.run(
                ["node", str(self.bridge), command],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=str(self.bridge.parent),
            )
        except subprocess.TimeoutExpired as exc:
            raise KnowledgeError(f"pi-knowledge bridge timed out ({self.timeout}s)") from exc
        line = next((strip for strip in reversed(proc.stdout.splitlines()) if strip.strip()), "")
        try:
            response = json.loads(line or "{}")
        except ValueError as exc:
            raise KnowledgeError(
                f"bridge returned invalid JSON (exit {proc.returncode}); stderr: "
                f"{proc.stderr.strip()[:400]}"
            ) from exc
        if not response.get("ok"):
            raise KnowledgeError(str(response.get("error") or "unknown bridge error"))
        return response

    # ------------------------------------------------------------------
    # tool surface
    # ------------------------------------------------------------------
    def search(
        self,
        queries: list[str],
        category: str = "any",
        top_k: int = 1,
        mode: str | None = None,
        profile: str | None = None,
    ) -> list[KnowledgeHit]:
        response = self._call(
            "search",
            {
                "queries": queries,
                "category": category,
                "top_k": top_k,
                "mode": mode or os.environ.get("PI_KNOWLEDGE_SEARCH_MODE", "semantic_hybrid"),
                "profile": profile or "balanced",
            },
        )
        hits = [
            KnowledgeHit(
                content=hit["content"],
                file_path=hit["file_path"],
                file_type=hit["file_type"],
                kb_name=hit["kb_name"],
                score=float(hit["score"]),
                snippet=hit["snippet"],
                start_line=int(hit.get("start_line", 0)),
                end_line=int(hit.get("end_line", 0)),
                chunk_id=(hit.get("provenance") or {}).get("chunk_id"),
                match_reason=(hit.get("provenance") or {}).get("match_reason"),
            )
            for hit in response.get("results", [])
        ]
        if not hits:
            # pi-knowledge reports misses explicitly; never fill with weak results.
            known = response.get("warnings") or []
            raise KnowledgeError("; ".join(formatted for formatted in known) or "no results")
        return hits

    def list(self) -> list[dict[str, Any]]:
        return self._call("list", {})["kbs"]

    def add(self, source: str | Path, include_suggested_text: bool = False) -> dict[str, Any]:
        return self._call(
            "add",
            {"source": str(source), "include_suggested_text": include_suggested_text},
        )

    def update(self, target: str | None = None) -> dict[str, Any]:
        return self._call("update", {"target": target or self.name})

    def doctor(self) -> dict[str, Any]:
        return self._call("doctor", {})

    # ------------------------------------------------------------------
    # output shaping for the model
    # ------------------------------------------------------------------
    def format_cards(self, hits: list[KnowledgeHit], limit: int = 4000) -> str:
        block: list[str] = []
        total = 0
        for hit in hits:
            header = f"{hit.file_path} (score {hit.score:.3f}, {hit.match_reason or 'hybrid'})"
            body = hit.content or hit.snippet
            if total + len(body) + len(header) > limit:
                block.append("… [truncated: additional cards omitted]")
                break
            total += len(body) + len(header)
            block.append(f"{header}\n{body}")
        return "\n\n".join(block)
