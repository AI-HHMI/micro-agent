#!/usr/bin/env python3
"""Unified MCP Server for microscopy repository discovery.

Abstracts across EMPIAR, IDR, MICrONS, BioImage Archive, and OpenOrganelle
to provide a single interface for dataset discovery across all repositories.

Usage:
    python unified_server.py

Dependencies:
    pip install mcp httpx
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("microscopy-unified")

TIMEOUT = 30.0


@dataclass
class UnifiedResult:
    repository: str
    accession: str
    title: str
    description: str = ""
    organism: str = ""
    imaging_modality: str = ""
    data_formats: list[str] = field(default_factory=list)
    access_url: str = ""
    python_snippet: str = ""
    cross_references: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-repository search backends
# ---------------------------------------------------------------------------


async def _search_idr(client: httpx.AsyncClient, query: str, limit: int) -> list[UnifiedResult]:
    """Search IDR using MAPR endpoints and full-text search."""
    results = []

    # Try the search engine first
    try:
        resp = await client.get(
            "https://idr.openmicroscopy.org/searchengine/api/v1/resources/image/search/",
            params={"value": query, "limit": limit},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("results", {}).get("results", [])[:limit]:
                results.append(
                    UnifiedResult(
                        repository="IDR",
                        accession=str(item.get("id", "")),
                        title=item.get("name", ""),
                        description=str(item.get("key_values", "")),
                        access_url=f"https://idr.openmicroscopy.org/webclient/img_detail/{item.get('id', '')}/",
                    )
                )
    except Exception:
        pass

    # Also try MAPR gene search (query might be a gene name)
    try:
        resp = await client.get(
            "https://idr.openmicroscopy.org/mapr/api/gene/",
            params={"value": query, "limit": limit},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            for screen in data.get("screens", []):
                results.append(
                    UnifiedResult(
                        repository="IDR",
                        accession=f"screen-{screen['id']}",
                        title=screen.get("name", ""),
                        description=f"Screen matching gene '{query}'",
                        data_formats=["OME-TIFF", "OME-Zarr"],
                        access_url=f"https://idr.openmicroscopy.org/webclient/?show=screen-{screen['id']}",
                    )
                )
    except Exception:
        pass

    return results[:limit]


async def _search_empiar(client: httpx.AsyncClient, query: str, limit: int) -> list[UnifiedResult]:
    """Search EMPIAR — limited to known entry IDs since there's no search endpoint."""
    results = []

    # EMPIAR has no search API. We can only look up specific IDs.
    # If the query looks like an ID, try it.
    query_clean = query.replace("EMPIAR-", "").strip()
    if query_clean.isdigit():
        try:
            empiar_id = f"EMPIAR-{query_clean}"
            resp = await client.get(
                f"https://www.ebi.ac.uk/empiar/api/entry/{empiar_id}/",
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                entry = data.get(empiar_id, {})
                results.append(
                    UnifiedResult(
                        repository="EMPIAR",
                        accession=empiar_id,
                        title=entry.get("title", ""),
                        organism=entry.get("organism", {}).get("organism", ""),
                        imaging_modality=entry.get("experiment_type", ""),
                        data_formats=["MRC", "TIFF"],
                        access_url=f"https://www.ebi.ac.uk/empiar/entry/{empiar_id}/",
                    )
                )
        except Exception:
            pass

    return results[:limit]


async def _search_bia(client: httpx.AsyncClient, query: str, limit: int) -> list[UnifiedResult]:
    """Search BioImage Archive via BioStudies search."""
    results = []

    try:
        resp = await client.get(
            "https://www.ebi.ac.uk/biostudies/BioImages/api/v1/search",
            params={"query": query, "limit": limit},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            for hit in data.get("hits", [])[:limit]:
                results.append(
                    UnifiedResult(
                        repository="BioImage Archive",
                        accession=hit.get("accession", ""),
                        title=hit.get("title", ""),
                        description=hit.get("content", ""),
                        data_formats=["OME-Zarr", "OME-TIFF"],
                        access_url=f"https://www.ebi.ac.uk/bioimage-archive/galleries/{hit.get('accession', '')}",
                    )
                )
    except Exception:
        pass

    return results[:limit]


async def _search_openorganelle(client: httpx.AsyncClient, query: str, limit: int) -> list[UnifiedResult]:
    """Search OpenOrganelle — searches the local catalog since there's no API."""
    # Import the catalog from openorganelle_server
    from openorganelle_server import KNOWN_DATASETS

    query_lower = query.lower()
    results = []

    for ds in KNOWN_DATASETS:
        searchable = json.dumps(ds).lower()
        if query_lower in searchable:
            results.append(
                UnifiedResult(
                    repository="OpenOrganelle",
                    accession=ds["id"],
                    title=ds["title"],
                    organism=ds["organism"],
                    imaging_modality=ds["imaging"],
                    data_formats=["N5", "Zarr"],
                    access_url=f"https://openorganelle.janelia.org/datasets/{ds['id']}",
                    python_snippet=(
                        f"import fibsem_tools as fst\n"
                        f"data = fst.read_xarray('s3://janelia-cosem-datasets/{ds['id']}/em/fibsem-uint8/s0', "
                        f"storage_options={{'anon': True}})"
                    ),
                )
            )

    return results[:limit]


# ---------------------------------------------------------------------------
# Unified tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_all(query: str, limit: int = 5) -> str:
    """Search across all microscopy repositories simultaneously.

    Fans out the query to IDR, EMPIAR, BioImage Archive, and OpenOrganelle
    in parallel. MICrONS is excluded by default since it requires auth.

    Args:
        query: Search query — can be a gene name, organism, keyword, or accession ID
        limit: Max results per repository
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            _search_idr(client, query, limit),
            _search_empiar(client, query, limit),
            _search_bia(client, query, limit),
            _search_openorganelle(client, query, limit),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    errors = []
    repo_names = ["IDR", "EMPIAR", "BioImage Archive", "OpenOrganelle"]
    for repo_name, result in zip(repo_names, results):
        if isinstance(result, Exception):
            errors.append({"repository": repo_name, "error": str(result)})
        else:
            all_results.extend(result)

    return json.dumps(
        {
            "query": query,
            "total_results": len(all_results),
            "results": [asdict(r) for r in all_results],
            "errors": errors if errors else None,
            "note": "MICrONS excluded (requires CAVE auth token). Use search_microns separately.",
        },
        indent=2,
    )


@mcp.tool()
async def search_by_organism_all(organism: str, limit: int = 5) -> str:
    """Search for datasets by organism across all repositories.

    Args:
        organism: Organism name (e.g., "Homo sapiens", "Drosophila melanogaster")
        limit: Max results per repository
    """
    async with httpx.AsyncClient() as client:
        tasks = []

        # IDR MAPR organism search
        async def idr_organism():
            results = []
            resp = await client.get(
                "https://idr.openmicroscopy.org/mapr/api/organism/",
                params={"value": organism, "limit": limit},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                for cat in ["screens", "projects"]:
                    for item in data.get(cat, []):
                        results.append(
                            UnifiedResult(
                                repository="IDR",
                                accession=f"{cat.rstrip('s')}-{item['id']}",
                                title=item.get("name", ""),
                                organism=organism,
                            )
                        )
            return results

        # OpenOrganelle catalog search
        async def oo_organism():
            return await _search_openorganelle(client, organism, limit)

        # BIA search
        async def bia_organism():
            return await _search_bia(client, organism, limit)

        all_tasks = [idr_organism(), oo_organism(), bia_organism()]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

    all_results = []
    for result in results:
        if not isinstance(result, Exception):
            all_results.extend(result)

    return json.dumps(
        {
            "organism": organism,
            "total_results": len(all_results),
            "results": [asdict(r) for r in all_results],
        },
        indent=2,
    )


@mcp.tool()
async def compare_repositories() -> str:
    """Get a comparison of API capabilities across all 5 microscopy repositories.

    Returns a structured overview useful for understanding what each repo
    can and cannot do programmatically.
    """
    return json.dumps(
        {
            "repositories": [
                {
                    "name": "EMPIAR",
                    "has_rest_api": True,
                    "has_search": False,
                    "auth_required": False,
                    "python_client": "empiarreader",
                    "data_formats": ["MRC", "TIFF", "DM4", "EER"],
                    "strengths": ["Entry lookup", "EMDB cross-references"],
                    "gaps": ["No search/filter", "No enumeration", "No file-level metadata"],
                },
                {
                    "name": "IDR",
                    "has_rest_api": True,
                    "has_search": True,
                    "auth_required": False,
                    "python_client": "idr-py",
                    "data_formats": ["OME-TIFF", "OME-Zarr", "JPEG"],
                    "strengths": ["Rich OMERO API", "MAPR queries (gene/organism/phenotype)", "Search engine"],
                    "gaps": ["No raw pixel REST access", "No original file download via web"],
                },
                {
                    "name": "MICrONS",
                    "has_rest_api": True,
                    "has_search": True,
                    "auth_required": True,
                    "python_client": "caveclient",
                    "data_formats": ["Neuroglancer Precomputed", "DataFrames"],
                    "strengths": ["Excellent Python client", "Rich annotation queries", "337M synapses"],
                    "gaps": ["Auth required for public data", "No spatial bbox queries", "No streaming"],
                },
                {
                    "name": "BioImage Archive",
                    "has_rest_api": True,
                    "has_search": True,
                    "auth_required": False,
                    "python_client": "biostudies-client",
                    "data_formats": ["OME-Zarr", "OME-TIFF"],
                    "strengths": ["BioStudies search", "S3 access", "REMBI metadata"],
                    "gaps": ["Split API surface", "Image-level API not public", "Poor filter docs"],
                },
                {
                    "name": "OpenOrganelle",
                    "has_rest_api": False,
                    "has_search": False,
                    "auth_required": False,
                    "python_client": "fibsem-tools",
                    "data_formats": ["N5", "Zarr", "Neuroglancer Precomputed"],
                    "strengths": ["Public S3 access", "High-resolution FIB-SEM", "Organelle segmentations"],
                    "gaps": ["No API at all", "No programmatic discovery", "Fragmented metadata"],
                },
            ]
        },
        indent=2,
    )


@mcp.tool()
async def get_access_code(repository: str, accession: str) -> str:
    """Generate Python code to access a dataset from any repository.

    Args:
        repository: Repository name ("EMPIAR", "IDR", "MICrONS", "BioImage Archive", "OpenOrganelle")
        accession: Dataset accession/ID in the given repository
    """
    snippets = {
        "EMPIAR": (
            f"# EMPIAR: {accession}\n"
            f"import requests\n"
            f"resp = requests.get('https://www.ebi.ac.uk/empiar/api/entry/{accession}/')\n"
            f"entry = resp.json()['{accession}']\n"
            f"print(entry['title'])\n"
            f"\n"
            f"# For data access, use empiarreader:\n"
            f"# pip install empiarreader\n"
            f"# from empiarreader import EmpiarSource\n"
            f"# source = EmpiarSource('{accession}')"
        ),
        "IDR": (
            f"# IDR: {accession}\n"
            f"import requests\n"
            f"# Get image metadata\n"
            f"resp = requests.get('https://idr.openmicroscopy.org/webgateway/imgData/{accession}/')\n"
            f"meta = resp.json()\n"
            f"print(meta['meta']['imageName'])\n"
            f"\n"
            f"# For OME-Zarr access:\n"
            f"# pip install ome-zarr\n"
            f"# import ome_zarr\n"
            f"# Check: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/{accession}.zarr/"
        ),
        "MICrONS": (
            f"# MICrONS: {accession}\n"
            f"# pip install caveclient cloud-volume\n"
            f"from caveclient import CAVEclient\n"
            f"client = CAVEclient('minnie65_public')\n"
            f"# client.auth.setup_token(make_new=True)  # first time only\n"
            f"\n"
            f"# Query cell types:\n"
            f"ct = client.materialize.query_table('aibs_metamodel_celltypes_v661', limit=10)\n"
            f"\n"
            f"# Access EM volume:\n"
            f"from cloudvolume import CloudVolume\n"
            f"vol = CloudVolume(client.info.image_source(), mip=0, use_https=True)"
        ),
        "BioImage Archive": (
            f"# BioImage Archive: {accession}\n"
            f"import requests\n"
            f"resp = requests.get('https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}')\n"
            f"study = resp.json()\n"
            f"print(study['section']['attributes'])\n"
            f"\n"
            f"# For OME-Zarr S3 access:\n"
            f"# aws --endpoint-url https://uk1s3.embassy.ebi.ac.uk --no-sign-request \\\n"
            f"#   s3 ls s3://bia-integrator-data/{accession}/"
        ),
        "OpenOrganelle": (
            f"# OpenOrganelle: {accession}\n"
            f"# pip install fibsem-tools\n"
            f"import fibsem_tools as fst\n"
            f"data = fst.read_xarray(\n"
            f"    's3://janelia-cosem-datasets/{accession}/em/fibsem-uint8/s0',\n"
            f"    storage_options={{'anon': True}}\n"
            f")\n"
            f"print(data)  # lazy dask-backed xarray with coords"
        ),
    }

    snippet = snippets.get(repository)
    if snippet:
        return json.dumps({"repository": repository, "accession": accession, "python_code": snippet}, indent=2)
    else:
        return json.dumps({"error": f"Unknown repository: {repository}. Use one of: {list(snippets.keys())}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
