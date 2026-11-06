"""Small error types for the optional knowledge tool."""
from __future__ import annotations


class PiAgentError(RuntimeError):
    """Base error for the agent runtime."""


class KnowledgeError(PiAgentError):
    """The knowledge tool is unavailable, misconfigured, or has no results."""


def format_knowledge_failure(exc: Exception) -> str:
    """A stable, readable message the model can treat as data."""
    detail = str(exc).strip() or exc.__class__.__name__
    return (
        "KNOWLEDGE UNAVAILABLE: " + detail + "; decide without knowledge cards this turn."
    )
