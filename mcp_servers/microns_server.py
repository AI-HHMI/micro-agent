#!/usr/bin/env python3
"""MCP Server for MICrONS (Allen Institute connectomics) discovery.

Wraps the CAVE (Connectome Annotation Versioning Engine) REST API to expose
MICrONS annotation tables, cell types, synapses, and materialization queries.

Note: MICrONS requires token authentication even for public data. Set the
CAVE_TOKEN environment variable, or run `caveclient` token setup first.

Usage:
    CAVE_TOKEN=your_token python microns_server.py

Dependencies:
    pip install mcp httpx
"""

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("microns-explorer")

CAVE_BASE = "https://global.daf-apis.com"
DATASTACK = "minnie65_public"
TIMEOUT = 60.0


def _headers() -> dict:
    token = os.environ.get("CAVE_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


@mcp.tool()
async def list_annotation_tables() -> str:
    """List all available annotation tables in the MICrONS dataset.

    Key tables include nucleus_detection_v0, synapses_pni_2,
    aibs_metamodel_celltypes_v661, proofreading_status_public_release.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{CAVE_BASE}/annotation/api/v2/aligned_volume/{DATASTACK}/table",
            headers=_headers(),
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


@mcp.tool()
async def get_table_metadata(table_name: str) -> str:
    """Get metadata (schema, description, row count) for an annotation table.

    Args:
        table_name: Table name (e.g., "nucleus_detection_v0", "synapses_pni_2")
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{CAVE_BASE}/annotation/api/v2/aligned_volume/{DATASTACK}/table/{table_name}",
            headers=_headers(),
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


@mcp.tool()
async def query_cell_types(cell_type: str = "", limit: int = 25) -> str:
    """Query the cell type classification table.

    Returns neurons classified by the AIBS metamodel with their cell type
    labels, positions, and root IDs.

    Args:
        cell_type: Filter by cell type label (e.g., "excitatory", "inhibitory").
                   Leave empty to get all types.
        limit: Max rows to return
    """
    table = "aibs_metamodel_celltypes_v661"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = {
            "table": table,
            "limit": limit,
        }
        if cell_type:
            payload["filter_equal_dict"] = {"cell_type": cell_type}

        resp = await client.post(
            f"{CAVE_BASE}/materialize/api/v3/datastack/{DATASTACK}/query",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


@mcp.tool()
async def query_synapses(
    pre_root_id: int | None = None,
    post_root_id: int | None = None,
    limit: int = 100,
) -> str:
    """Query the synapse table for connections between neurons.

    WARNING: The full table has ~337M rows. Always filter by pre or post
    root ID to avoid timeouts.

    Args:
        pre_root_id: Presynaptic neuron root ID (filter)
        post_root_id: Postsynaptic neuron root ID (filter)
        limit: Max rows to return
    """
    if not pre_root_id and not post_root_id:
        return json.dumps(
            {
                "error": "Must provide pre_root_id or post_root_id to filter. "
                "Unfiltered queries on 337M rows will time out."
            }
        )

    table = "synapses_pni_2"
    filter_dict = {}
    if pre_root_id:
        filter_dict["pre_pt_root_id"] = pre_root_id
    if post_root_id:
        filter_dict["post_pt_root_id"] = post_root_id

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = {
            "table": table,
            "limit": limit,
            "filter_equal_dict": filter_dict,
        }
        resp = await client.post(
            f"{CAVE_BASE}/materialize/api/v3/datastack/{DATASTACK}/query",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


@mcp.tool()
async def get_datastack_info() -> str:
    """Get info about the MICrONS datastack (data sources, versions, links)."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{CAVE_BASE}/info/api/v2/datastack/full/{DATASTACK}",
            headers=_headers(),
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
