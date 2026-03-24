#!/usr/bin/env python3
"""MCP Server for BioImage Archive (BIA) discovery.

BIA uses the BioStudies API for study-level metadata. Image-level metadata
is accessed via the BIA Integrator (internal tooling) or direct S3 access.

Usage:
    python bia_server.py

Dependencies:
    pip install mcp httpx
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bia-explorer")

BIOSTUDIES_BASE = "https://www.ebi.ac.uk/biostudies/api/v1"
BIA_S3_ENDPOINT = "https://uk1s3.embassy.ebi.ac.uk"
TIMEOUT = 30.0


@mcp.tool()
async def get_study(accession: str) -> str:
    """Get full metadata for a BioImage Archive study.

    Returns the complete BioStudies submission including sections,
    file lists, attributes, and links.

    Args:
        accession: BIA accession number (e.g., "S-BIAD570")
    """
    accession = accession.strip().upper()
    if not accession.startswith("S-BIAD"):
        accession = f"S-BIAD{accession}"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{BIOSTUDIES_BASE}/studies/{accession}")
        resp.raise_for_status()
        data = resp.json()

        # Extract key attributes from nested structure
        attrs = {}
        for attr in data.get("attributes", []):
            attrs[attr.get("name", "")] = attr.get("value", "")

        section = data.get("section", {})
        section_attrs = {}
        for attr in section.get("attributes", []):
            section_attrs[attr.get("name", "")] = attr.get("value", "")

        return json.dumps(
            {
                "accession": data.get("accno", accession),
                "title": section_attrs.get("Title", attrs.get("Title", "")),
                "release_date": data.get("rtime"),
                "type": data.get("type", ""),
                "attributes": section_attrs,
                "links": [
                    {"url": link.get("url", ""), "description": link.get("attributes", [{}])[0].get("value", "")}
                    for link in section.get("links", [])
                    if isinstance(link, dict)
                ],
                "file_count": len(section.get("files", [])),
                "portal_url": f"https://www.ebi.ac.uk/bioimage-archive/galleries/{accession}",
            },
            indent=2,
        )


@mcp.tool()
async def get_study_info(accession: str) -> str:
    """Get concise study info including FTP download link.

    Args:
        accession: BIA accession number (e.g., "S-BIAD570")
    """
    accession = accession.strip().upper()
    if not accession.startswith("S-BIAD"):
        accession = f"S-BIAD{accession}"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{BIOSTUDIES_BASE}/studies/{accession}/info")
        resp.raise_for_status()
        data = resp.json()

        return json.dumps(
            {
                "accession": accession,
                "ftp_link": data.get("ftpLink", ""),
                "release_date": data.get("released", ""),
                "files": data.get("files", 0),
                "s3_access": f"aws --endpoint-url {BIA_S3_ENDPOINT} s3 ls s3://bia-integrator-data/{accession}/",
            },
            indent=2,
        )


@mcp.tool()
async def search_studies(query: str, limit: int = 10) -> str:
    """Search BioImage Archive studies via BioStudies search.

    Note: This searches across all BioStudies, not just BIA. Results are
    filtered to the BioImages collection where possible.

    Args:
        query: Free-text search query (e.g., "mitochondria", "fluorescence")
        limit: Max results to return
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(
            "https://www.ebi.ac.uk/biostudies/BioImages/api/v1/search",
            params={"query": query, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits", [])
        results = []
        for hit in hits:
            results.append(
                {
                    "accession": hit.get("accession", ""),
                    "title": hit.get("title", ""),
                    "author": hit.get("author", ""),
                    "release_date": hit.get("release_date", ""),
                    "links": hit.get("links", 0),
                    "files": hit.get("files", 0),
                }
            )
        return json.dumps(
            {
                "total_hits": data.get("totalHits", 0),
                "results": results,
            },
            indent=2,
        )


@mcp.tool()
async def get_ome_zarr_url(accession: str) -> str:
    """Get the S3 URL for OME-Zarr cloud-optimized data access.

    Many BIA datasets are available as OME-Zarr on EBI's S3, enabling
    lazy/streaming access without full download.

    Args:
        accession: BIA accession number (e.g., "S-BIAD570")
    """
    accession = accession.strip().upper()
    if not accession.startswith("S-BIAD"):
        accession = f"S-BIAD{accession}"

    return json.dumps(
        {
            "accession": accession,
            "s3_endpoint": BIA_S3_ENDPOINT,
            "s3_bucket": "bia-integrator-data",
            "s3_prefix": accession,
            "python_example": (
                f"import zarr\n"
                f"import s3fs\n"
                f"fs = s3fs.S3FileSystem(anon=True, client_kwargs={{'endpoint_url': '{BIA_S3_ENDPOINT}'}})\n"
                f"store = s3fs.S3Map(root='bia-integrator-data/{accession}/', s3=fs)\n"
                f"z = zarr.open(store, mode='r')"
            ),
            "cli_command": f"aws --endpoint-url {BIA_S3_ENDPOINT} --no-sign-request s3 ls s3://bia-integrator-data/{accession}/",
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
