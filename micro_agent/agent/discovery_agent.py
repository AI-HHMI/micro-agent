"""LLM-driven dataset discovery agent.

Runs a tool-use loop where an LLM plans which sources to scan, extracts
metadata from results, validates accessibility, and saves candidates to
the catalog with validation status flags.

Usage:
    python -m micro_agent.agent.discovery_agent [--focus "light sheet datasets"]
    python -m micro_agent.agent.discovery_agent --provider litellm --model gpt-4o
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from micro_agent.agent.llm import AgentLLM, AgentMessage
from micro_agent.agent.tools import ALL_TOOLS, ToolExecutor
from micro_agent.discover import DiscoveredDataset
from micro_agent.registry import Registry

SYSTEM_PROMPT = """\
You are a microscopy dataset discovery agent. Your job is to find publicly
accessible microscopy datasets (both electron microscopy and fluorescence)
from scientific data repositories.

You have access to tools that let you:
1. Scan specific repositories (OpenOrganelle, EMPIAR, IDR, BioImage Archive,
   Allen Institute, Human Protein Atlas, Cell Image Library, Zenodo)
2. Search the web for newly published datasets
3. Fetch and read web pages for metadata extraction
4. Validate that datasets are accessible
5. Check if datasets already exist in the registry
6. Save new dataset candidates

For each discovery cycle:
1. Plan which repositories to scan based on the focus query
2. Scan the most relevant repositories
3. For promising results, validate accessibility
4. Extract and normalize metadata (organism, imaging modality, organelles, etc.)
5. Save validated candidates

Prioritize datasets that:
- Have paired raw + segmentation data
- Are freely accessible without authentication
- Have clear metadata (organism, imaging modality, voxel size)
- Include fluorescence/multi-channel data when the focus mentions it

Be efficient — don't scan all repositories if the focus is narrow.
Always validate before saving. Report what you found at the end.
"""

MAX_TURNS = 20  # Safety limit on agent loop iterations


class DiscoveryAgent:
    """LLM-driven discovery agent that finds microscopy datasets."""

    def __init__(
        self,
        llm: AgentLLM | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.llm = llm or AgentLLM()
        self.registry = registry or Registry()
        self.executor = ToolExecutor(self.registry)

    async def run_cycle(self, focus: str = "") -> list[DiscoveredDataset]:
        """Run one discovery cycle.

        Args:
            focus: Optional focus query to guide the agent (e.g.,
                "find recent light sheet datasets", "fluorescence cell atlas").
                If empty, the agent scans all sources broadly.

        Returns:
            List of discovered dataset candidates.
        """
        user_prompt = (
            f"Run a discovery cycle. Focus: {focus}"
            if focus
            else "Run a broad discovery cycle across all available sources."
        )

        messages: list[AgentMessage] = [
            AgentMessage(role="user", content=user_prompt),
        ]

        print(f"Discovery agent starting cycle (focus: {focus or 'broad'})...")

        for turn in range(MAX_TURNS):
            # Get LLM response
            response = await self.llm.chat_with_tools(
                messages=messages,
                tools=ALL_TOOLS,
                system=SYSTEM_PROMPT,
            )

            messages.append(response)

            # If no tool calls, the agent is done
            if not response.tool_calls:
                if response.content:
                    print(f"\nAgent summary: {response.content[:500]}")
                break

            # Execute tool calls
            for tc in response.tool_calls:
                print(f"  [{turn + 1}] Calling {tc.name}({json.dumps(tc.arguments)[:100]}...)")
                result = await self.executor.execute(tc.name, tc.arguments)
                messages.append(AgentMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                ))

        candidates = self.executor.candidates
        print(f"\nDiscovery cycle complete: {len(candidates)} candidates found")
        return candidates

    async def run_and_save(
        self,
        focus: str = "",
        output_path: str = "discovered_datasets.json",
    ) -> list[DiscoveredDataset]:
        """Run a discovery cycle and save results to JSON."""
        candidates = await self.run_cycle(focus)

        # Merge with any existing discoveries
        output_file = Path(output_path)
        existing: list[dict] = []
        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)

        existing_ids = {e["id"] for e in existing}
        new_entries = [asdict(c) for c in candidates if c.id not in existing_ids]

        all_entries = existing + new_entries
        with open(output_file, "w") as f:
            json.dump(all_entries, f, indent=2)

        print(f"Saved {len(new_entries)} new entries to {output_path}")
        print(f"Total entries in file: {len(all_entries)}")
        return candidates


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agentic microscopy dataset discovery")
    parser.add_argument("--focus", default="", help="Focus query for the discovery cycle")
    parser.add_argument("--output", default="discovered_datasets.json", help="Output JSON path")
    parser.add_argument("--provider", default=None, help="LLM provider (anthropic or litellm)")
    parser.add_argument("--model", default=None, help="LLM model name")
    args = parser.parse_args()

    llm = AgentLLM(provider=args.provider, model=args.model)
    agent = DiscoveryAgent(llm=llm)
    await agent.run_and_save(focus=args.focus, output_path=args.output)


if __name__ == "__main__":
    asyncio.run(_main())
