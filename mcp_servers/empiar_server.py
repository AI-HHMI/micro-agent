#!/usr/bin/env python3
"""MCP Server for EMPIAR (Electron Microscopy Public Image Archive) discovery.

EMPIAR's REST API is limited — it supports entry lookup by ID and EMDB
cross-reference but has no search or enumeration endpoints. This MCP server
wraps the available endpoints and adds a local metadata cache for discovery.

Usage:
    python empiar_server.py

Dependencies:
    pip install mcp httpx
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("empiar-explorer")

EMPIAR_BASE = "https://www.ebi.ac.uk/empiar/api"
TIMEOUT = 30.0


@mcp.tool()
async def get_entry(entry_id: str) -> str:
    """Get metadata for a single EMPIAR entry.

    Returns title, authors, organism, experimental method, image sets,
    and cross-references (EMDB, PDB).

    Args:
        entry_id: EMPIAR entry ID (e.g., "10310" or "EMPIAR-10310")
    """
    # Normalize ID format
    entry_id = entry_id.replace("EMPIAR-", "").strip()
    empiar_id = f"EMPIAR-{entry_id}"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{EMPIAR_BASE}/entry/{empiar_id}/")
        resp.raise_for_status()
        data = resp.json()

        entry = data.get(empiar_id, {})
        return json.dumps(
            {
                "id": empiar_id,
                "title": entry.get("title", ""),
                "authors": entry.get("authors", ""),
                "organism": entry.get("organism", {}).get("organism", ""),
                "experiment_type": entry.get("experiment_type", ""),
                "release_date": entry.get("release_date", ""),
                "dataset_size": entry.get("dataset_size", ""),
                "imagesets": [
                    {
                        "name": iset.get("name", ""),
                        "directory": iset.get("directory", ""),
                        "num_images": iset.get("num_images_or_tilt_series", ""),
                        "data_format": iset.get("data_format", ""),
                        "image_width": iset.get("image_width", ""),
                        "image_height": iset.get("image_height", ""),
                    }
                    for iset in entry.get("imagesets", [])
                ],
                "cross_references": entry.get("cross_references", {}),
                "empiar_url": f"https://www.ebi.ac.uk/empiar/entry/{empiar_id}/",
            },
            indent=2,
        )


@mcp.tool()
async def get_entries_batch(entry_ids: list[str]) -> str:
    """Get metadata for multiple EMPIAR entries in one call.

    Uses the POST endpoint to fetch batch entry metadata.

    Args:
        entry_ids: List of EMPIAR entry IDs (e.g., ["10310", "10940"])
    """
    normalized = [f"EMPIAR-{eid.replace('EMPIAR-', '').strip()}" for eid in entry_ids]

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{EMPIAR_BASE}/entry/",
            json={"entry_ids": normalized},
        )
        resp.raise_for_status()
        data = resp.json()

        results = {}
        for empiar_id, entry in data.items():
            results[empiar_id] = {
                "title": entry.get("title", ""),
                "organism": entry.get("organism", {}).get("organism", ""),
                "experiment_type": entry.get("experiment_type", ""),
                "release_date": entry.get("release_date", ""),
            }
        return json.dumps(results, indent=2)


@mcp.tool()
async def get_by_emdb_id(emdb_id: str) -> str:
    """Find EMPIAR entries linked to an EMDB (Electron Microscopy Data Bank) entry.

    Useful for finding raw data (in EMPIAR) associated with a reconstructed
    map (in EMDB).

    Args:
        emdb_id: EMDB entry ID (e.g., "EMD-1234" or "1234")
    """
    emdb_id = emdb_id.replace("EMD-", "").strip()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{EMPIAR_BASE}/emdb_entry/EMD-{emdb_id}/")
        resp.raise_for_status()
        data = resp.json()

        results = []
        for empiar_id, entry in data.items():
            results.append(
                {
                    "empiar_id": empiar_id,
                    "title": entry.get("title", ""),
                    "organism": entry.get("organism", {}).get("organism", ""),
                    "experiment_type": entry.get("experiment_type", ""),
                }
            )
        return json.dumps(results, indent=2)


@mcp.tool()
async def get_recent_citations() -> str:
    """Get the latest publications citing EMPIAR.

    Returns up to 5 recent citations from Europe PMC.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{EMPIAR_BASE}/empiar_citations/")
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
