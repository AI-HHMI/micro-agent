"""Selectable LLM backend for the discovery agent.

Supports:
- Anthropic SDK (Claude) — native tool-use
- OpenAI-compatible APIs (GPT, Ollama, Together, Groq) — via litellm

Configure via environment variables:
    TRAILHEAD_LLM_PROVIDER: "anthropic" or "litellm" (default: "anthropic")
    TRAILHEAD_LLM_MODEL: model name (default: "claude-sonnet-4-20250514")
    ANTHROPIC_API_KEY: for Anthropic provider
    OPENAI_API_KEY: for OpenAI-compatible providers via litellm
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
    """Unified LLM interface for the discovery agent.

    Args:
        provider: "anthropic" or "litellm"
        model: Model name/ID
        api_key: API key (falls back to env vars)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider or os.environ.get("TRAILHEAD_LLM_PROVIDER", "anthropic")
        self.model = model or os.environ.get("TRAILHEAD_LLM_MODEL", "claude-sonnet-4-20250514")
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
        if self.provider == "anthropic":
            return await self._chat_anthropic(messages, tools, system)
        elif self.provider == "litellm":
            return await self._chat_litellm(messages, tools, system)
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'anthropic' or 'litellm'.")

    async def _chat_anthropic(
        self,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        system: str,
    ) -> AgentMessage:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install micro_agent[agent]"
            )

        client = anthropic.AsyncAnthropic(
            api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        # Convert messages to Anthropic format
        api_messages = []
        for msg in messages:
            if msg.role == "user":
                api_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                api_messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })

        # Convert tools to Anthropic format
        api_tools = []
        for tool in tools:
            api_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        response = await client.messages.create(**kwargs)

        # Parse response
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return AgentMessage(
            role="assistant",
            content="\n".join(text_parts),
            tool_calls=tool_calls,
        )

    async def _chat_litellm(
        self,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        system: str,
    ) -> AgentMessage:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm package required. Install with: pip install micro_agent[agent]"
            )

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
