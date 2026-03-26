#!/usr/bin/env python3
"""MCP Server for IDR (Image Data Resource) discovery.

IDR is a public, read-only OMERO server hosting curated microscopy reference
datasets. This MCP server exposes IDR's REST API layers (OMERO JSON API,
Webclient API, MAPR) as tools that Claude can call for dataset discovery.

Usage:
    python idr_server.py                          # stdio transport (default)
    mcp dev idr_server.py                         # MCP inspector
    Add to claude_desktop_config.json for Claude Desktop integration.

Dependencies:
    pip install mcp httpx
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("idr-explorer")

IDR_BASE = "https://idr.openmicroscopy.org"
TIMEOUT = 30.0


@mcp.tool()
async def search_by_gene(gene_name: str, limit: int = 10) -> str:
    """Search IDR for screens/projects associated with a gene name.

    Uses the MAPR (Map Annotation Reverse Query) endpoint to find studies
    where images are annotated with the given gene symbol.

    Args:
        gene_name: Gene symbol (e.g., "BRCA1", "TP53", "CDK1")
        limit: Max results to return
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{IDR_BASE}/mapr/api/gene/",
            params={"value": gene_name, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for screen in data.get("screens", []):
            results.append(
                {
                    "type": "screen",
                    "id": screen["id"],
                    "name": screen.get("name", ""),
                    "url": f"{IDR_BASE}/webclient/?show=screen-{screen['id']}",
                }
            )
        for project in data.get("projects", []):
            results.append(
                {
                    "type": "project",
                    "id": project["id"],
                    "name": project.get("name", ""),
                    "url": f"{IDR_BASE}/webclient/?show=project-{project['id']}",
                }
            )
        return json.dumps(results, indent=2)


@mcp.tool()
async def search_by_organism(organism: str, limit: int = 20) -> str:
    """Find IDR studies by organism name.

    Args:
        organism: Organism name (e.g., "Homo sapiens", "Drosophila melanogaster")
        limit: Max results to return
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{IDR_BASE}/mapr/api/organism/",
            params={"value": organism, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for category in ["screens", "projects"]:
            for item in data.get(category, []):
                results.append(
                    {
                        "type": category.rstrip("s"),
                        "id": item["id"],
                        "name": item.get("name", ""),
                    }
                )
        return json.dumps(results, indent=2)


@mcp.tool()
async def search_by_phenotype(phenotype: str, limit: int = 10) -> str:
    """Find IDR studies by phenotype annotation.

    Args:
        phenotype: Phenotype term (e.g., "mitotic", "apoptosis")
        limit: Max results to return
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{IDR_BASE}/mapr/api/phenotype/",
            params={"value": phenotype, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for category in ["screens", "projects"]:
            for item in data.get(category, []):
                results.append(
                    {
                        "type": category.rstrip("s"),
                        "id": item["id"],
                        "name": item.get("name", ""),
                    }
                )
        return json.dumps(results, indent=2)


@mcp.tool()
async def list_studies(limit: int = 25, offset: int = 0) -> str:
    """List available studies (screens and projects) in IDR.

    IDR organizes data as either Screens (high-content screening) or
    Projects (other imaging experiments). Each contains datasets/plates
    which contain images.

    Args:
        limit: Max results per category
        offset: Pagination offset
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        screens_resp = await client.get(
            f"{IDR_BASE}/api/v0/m/screens/",
            params={"limit": limit, "offset": offset},
        )
        projects_resp = await client.get(
            f"{IDR_BASE}/api/v0/m/projects/",
            params={"limit": limit, "offset": offset},
        )

        screens = screens_resp.json()
        projects = projects_resp.json()

        return json.dumps(
            {
                "screens": [
                    {"id": s["@id"], "name": s.get("Name", "")}
                    for s in screens.get("data", [])
                ],
                "projects": [
                    {"id": p["@id"], "name": p.get("Name", "")}
                    for p in projects.get("data", [])
                ],
                "total_screens": screens.get("meta", {}).get("totalCount", 0),
                "total_projects": projects.get("meta", {}).get("totalCount", 0),
            },
            indent=2,
        )


@mcp.tool()
async def get_dataset_images(dataset_id: int, limit: int = 50) -> str:
    """List images in an IDR dataset with dimensional metadata.

    Args:
        dataset_id: IDR dataset ID
        limit: Max images to return
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{IDR_BASE}/api/v0/m/datasets/{dataset_id}/images/",
            params={"limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        images = []
        for img in data.get("data", []):
            pixels = img.get("Pixels", {})
            images.append(
                {
                    "id": img["@id"],
                    "name": img.get("Name", ""),
                    "sizeX": pixels.get("SizeX"),
                    "sizeY": pixels.get("SizeY"),
                    "sizeZ": pixels.get("SizeZ"),
                    "sizeC": pixels.get("SizeC"),
                    "sizeT": pixels.get("SizeT"),
                    "pixel_type": pixels.get("Type"),
                }
            )
        return json.dumps(
            {
                "total": data.get("meta", {}).get("totalCount", 0),
                "images": images,
            },
            indent=2,
        )


@mcp.tool()
async def get_image_annotations(image_id: int) -> str:
    """Get key-value map annotations for an IDR image.

    Map annotations in IDR contain rich biological metadata: gene names,
    phenotypes, organism, cell line, compound treatments, etc.

    Args:
        image_id: IDR image ID
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            f"{IDR_BASE}/webclient/api/annotations/",
            params={"type": "map", "image": image_id},
        )
        resp.raise_for_status()
        data = resp.json()

        annotations = {}
        for ann in data.get("annotations", []):
            for kv in ann.get("values", []):
                annotations[kv[0]] = kv[1]
        return json.dumps(annotations, indent=2)


@mcp.tool()
async def get_image_details(image_id: int) -> str:
    """Get full metadata for a specific IDR image.

    Returns dimensions, channel info, pixel sizes, and rendering settings.

    Args:
        image_id: IDR image ID
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{IDR_BASE}/webgateway/imgData/{image_id}/")
        resp.raise_for_status()
        data = resp.json()

        return json.dumps(
            {
                "id": data.get("id"),
                "name": data.get("meta", {}).get("imageName"),
                "sizeX": data.get("size", {}).get("width"),
                "sizeY": data.get("size", {}).get("height"),
                "sizeZ": data.get("size", {}).get("z"),
                "sizeC": data.get("size", {}).get("c"),
                "sizeT": data.get("size", {}).get("t"),
                "pixel_size": data.get("pixel_size", {}),
                "channels": [
                    {"label": ch.get("label"), "color": ch.get("color")}
                    for ch in data.get("channels", [])
                ],
                "thumbnail_url": f"{IDR_BASE}/webgateway/render_thumbnail/{image_id}/",
                "viewer_url": f"{IDR_BASE}/webclient/img_detail/{image_id}/",
            },
            indent=2,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
