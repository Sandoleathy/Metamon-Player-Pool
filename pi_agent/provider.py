"""Pi agent construction and OpenCode Go provider wiring."""
from __future__ import annotations

from typing import Any, Sequence

from .config import AgentConfig


def opencode_provider(config: AgentConfig) -> Any:
    """An OpenAI-compatible provider pointed at OpenCode Go (or a custom root)."""
    from pi_python.providers import OpenAICompletionsProvider

    return OpenAICompletionsProvider(
        base_url=config.base_url,
        name="opencode-go",
        api_key=config.api_key,
    )


def opencode_model(provider: Any, config: AgentConfig) -> Any:
    return provider.model(
        config.model,
        context_window=config.context_window,
        max_tokens=config.max_tokens,
        reasoning=config.reasoning,
        name=config.model,
    )


def build_agent(
    config: AgentConfig,
    *,
    tools: Sequence[Any],
    system_prompt: str,
    provider: Any | None = None,
    model: Any | None = None,
) -> Any:
    """Create one Pi agent for one battle turn."""
    from pi_python import Agent, RunLimits

    provider = provider or opencode_provider(config)
    model = model or opencode_model(provider, config)
    return Agent(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        thinking_level=config.thinking,
        limits=RunLimits(
            max_model_requests=config.max_model_requests,
            max_tool_calls=config.max_tool_calls,
            run_timeout=config.run_timeout,
        ),
    )
