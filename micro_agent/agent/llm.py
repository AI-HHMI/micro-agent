"""LLM backend for the discovery agent.

Uses litellm, which routes to any supported provider (Anthropic, OpenAI,
Ollama, Together, Groq, etc.) based on the model name.

Configure via environment variables:
    MICRO_AGENT_LLM_MODEL: model name (default: "claude-sonnet-4-20250514")
    ANTHROPIC_API_KEY / OPENAI_API_KEY / etc.: provider API key, per the
        model's provider
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class AgentMessage:
    """A message in the agent conversation."""

    role: str  # "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""  # For tool result messages


@dataclass
class ToolDefinition:
    """A tool the LLM can call."""

    name: str
    description: str
    parameters: dict  # JSON Schema


class AgentLLM:
    """LLM interface for the discovery agent, backed by litellm.

    Args:
        model: Model name/ID (litellm routes to the right provider)
        api_key: API key (falls back to provider-specific env var)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("MICRO_AGENT_LLM_MODEL", "claude-sonnet-4-20250514")
        self._api_key = api_key

    async def chat_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        system: str = "",
    ) -> AgentMessage:
        """Send messages with tool definitions and get a response.

        Returns an AgentMessage that may contain text content, tool_calls, or both.
        """
        import litellm

        # Convert messages to OpenAI format
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "user":
                api_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                m: dict = {"role": "assistant"}
                if msg.content:
                    m["content"] = msg.content
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                api_messages.append(m)
            elif msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })

        # Convert tools to OpenAI format
        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = await litellm.acompletion(**kwargs)
        choice = response.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return AgentMessage(
            role="assistant",
            content=choice.content or "",
            tool_calls=tool_calls,
        )
