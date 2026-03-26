#!/usr/bin/env python3
"""MCP Server for agentic dataset discovery.

Exposes tools to trigger discovery cycles, list candidates, validate
datasets, and manage validation status — all accessible from Claude
Desktop or any MCP client.

Usage:
    python discovery_server.py

Dependencies:
    pip install mcp httpx trailhead[agent]
"""

import asyncio
import json
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("microscopy-discovery")


@mcp.tool()
async def discover_now(
    sources: str = "",
    focus_query: str = "",
    limit: int = 25,
    validate: bool = True,
) -> str:
    """Trigger a dataset discovery scan across microscopy repositories.

    Args:
        sources: Comma-separated list of sources to scan. Leave empty for all.
            Available: OpenOrganelle, EMPIAR, IDR, BioImage Archive, Allen, HPA,
            CellImageLibrary, Zenodo
        focus_query: Optional focus (e.g., "fluorescence cell atlas", "light sheet")
        limit: Max datasets per source
        validate: Whether to validate accessibility of discovered datasets
    """
    from trailhead.scanners import ALL_SCANNERS, run_all_scanners

    if sources:
        source_names = [s.strip() for s in sources.split(",")]
        scanners = [
            cls() for cls in ALL_SCANNERS
            if cls.name in source_names  # type: ignore[attr-defined]
        ]
        if not scanners:
            return json.dumps({
                "error": f"No matching scanners for: {source_names}",
                "available": [cls.name for cls in ALL_SCANNERS],  # type: ignore[attr-defined]
            })
    else:
        scanners = None  # Use all

    results = run_all_scanners(scanners=scanners, limit=limit, validate=validate)

    summary = {
        "total_discovered": len(results),
        "by_modality": {},
        "by_repository": {},
        "by_status": {},
        "samples": [asdict(r) for r in results[:5]],
    }

    for r in results:
        mc = r.modality_class or "unknown"
        summary["by_modality"][mc] = summary["by_modality"].get(mc, 0) + 1
        summary["by_repository"][r.repository] = summary["by_repository"].get(r.repository, 0) + 1
        summary["by_status"][r.validation_status] = summary["by_status"].get(r.validation_status, 0) + 1

    return json.dumps(summary, indent=2)


@mcp.tool()
async def discover_with_agent(
    focus: str = "",
    provider: str = "anthropic",
    model: str = "",
) -> str:
    """Run an LLM-driven discovery cycle that intelligently searches for datasets.

    The agent plans which sources to check, extracts metadata, validates
    accessibility, and returns candidates. Requires an API key for the
    chosen LLM provider.

    Args:
        focus: What to search for (e.g., "recent light sheet datasets",
            "fluorescence cell atlas", "new EM connectome data")
        provider: LLM provider — "anthropic" or "litellm"
        model: Model name (default: claude-sonnet-4-20250514 for anthropic)
    """
    from trailhead.agent.llm import AgentLLM
    from trailhead.agent.discovery_agent import DiscoveryAgent

    llm = AgentLLM(provider=provider, model=model or None)
    agent = DiscoveryAgent(llm=llm)
    candidates = await agent.run_cycle(focus=focus)

    return json.dumps({
        "candidates_found": len(candidates),
        "datasets": [asdict(c) for c in candidates[:10]],
    }, indent=2)


@mcp.tool()
async def list_candidates(
    status_filter: str = "",
    modality_filter: str = "",
    repository_filter: str = "",
) -> str:
    """List discovered dataset candidates from the catalog.

    Args:
        status_filter: Filter by validation_status (verified, failed, pending)
        modality_filter: Filter by modality_class (em, fluorescence)
        repository_filter: Filter by repository name
    """
    from trailhead.registry import Registry

    registry = Registry()
    entries = registry.search(
        validation_status=status_filter,
        modality_class=modality_filter,
        repository=repository_filter,
    )

    return json.dumps({
        "total": len(entries),
        "datasets": [
            {
                "id": e.id,
                "repository": e.repository,
                "title": e.title,
                "modality_class": e.modality_class,
                "validation_status": e.validation_status,
                "num_channels": e.num_channels,
                "organism": e.organism,
            }
            for e in entries[:50]
        ],
    }, indent=2)


@mcp.tool()
async def validate_entry(dataset_id: str) -> str:
    """Re-validate a specific dataset's accessibility.

    Args:
        dataset_id: The dataset ID to validate
    """
    from trailhead.registry import Registry
    from trailhead.discover import DiscoveredDataset
    from trailhead.validate import validate_dataset

    registry = Registry()
    matches = [e for e in registry.entries if e.id == dataset_id]
    if not matches:
        return json.dumps({"error": f"Dataset '{dataset_id}' not found in registry"})

    entry = matches[0]
    ds = DiscoveredDataset(
        id=entry.id,
        repository=entry.repository,
        title=entry.title,
        access_url=entry.access_url,
        data_format=entry.data_format,
    )

    result = await validate_dataset(ds)
    return json.dumps({
        "id": dataset_id,
        "validation_status": result.status,
        "accessible": result.accessible,
        "error": result.error,
    }, indent=2)


@mcp.tool()
async def search_fluorescence(
    fluorophore: str = "",
    organism: str = "",
    channel_name: str = "",
) -> str:
    """Search specifically for fluorescence microscopy datasets.

    Args:
        fluorophore: Filter by fluorophore name (e.g., "GFP", "DAPI")
        organism: Filter by organism
        channel_name: Filter by channel name
    """
    from trailhead.registry import Registry

    registry = Registry()
    entries = registry.search(
        modality_class="fluorescence",
        organism=organism,
    )

    # Additional filtering by fluorophore/channel
    if fluorophore:
        fl = fluorophore.lower()
        entries = [
            e for e in entries
            if any(fl in f.lower() for f in e.fluorophores)
            or any(fl in c.lower() for c in e.channel_names)
        ]

    if channel_name:
        cn = channel_name.lower()
        entries = [
            e for e in entries
            if any(cn in c.lower() for c in e.channel_names)
        ]

    return json.dumps({
        "total": len(entries),
        "datasets": [
            {
                "id": e.id,
                "repository": e.repository,
                "title": e.title,
                "num_channels": e.num_channels,
                "channel_names": e.channel_names,
                "fluorophores": e.fluorophores,
                "organism": e.organism,
            }
            for e in entries[:50]
        ],
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
