#!/usr/bin/env python3
"""MCP Server for OpenOrganelle (Janelia/CellMap) discovery.

OpenOrganelle has NO REST API. Data lives in public S3 buckets as N5/Zarr
with metadata in attributes.json files. This MCP server provides
programmatic discovery by listing S3 buckets and parsing dataset metadata.

Usage:
    python openorganelle_server.py

Dependencies:
    pip install mcp httpx
    Optional: pip install boto3 (for S3 listing)
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("openorganelle-explorer")

# Known S3 buckets (us-east-1, anonymous access)
DATASETS_BUCKET = "janelia-cosem-datasets"
NETWORKS_BUCKET = "janelia-cosem-networks"
PUBLICATIONS_BUCKET = "janelia-cosem-publications"

# Known datasets catalog (hardcoded since there's no API)
# This would ideally be built by crawling S3, but we bootstrap with
# known datasets from the web portal.
KNOWN_DATASETS = [
    {
        "id": "jrc_hela-2",
        "title": "HeLa cell (Interphase) #2",
        "organism": "Homo sapiens",
        "cell_type": "HeLa",
        "imaging": "FIB-SEM",
        "voxel_nm": [8, 8, 8],
        "organelles_segmented": ["mito", "er", "nucleus", "golgi", "vesicles", "microtubules"],
        "s3_path": f"s3://{DATASETS_BUCKET}/jrc_hela-2/",
    },
    {
        "id": "jrc_hela-3",
        "title": "HeLa cell (Interphase) #3",
        "organism": "Homo sapiens",
        "cell_type": "HeLa",
        "imaging": "FIB-SEM",
        "voxel_nm": [4, 4, 4],
        "organelles_segmented": ["mito", "er", "nucleus", "golgi"],
        "s3_path": f"s3://{DATASETS_BUCKET}/jrc_hela-3/",
    },
    {
        "id": "jrc_macrophage-2",
        "title": "Macrophage #2",
        "organism": "Homo sapiens",
        "cell_type": "macrophage",
        "imaging": "FIB-SEM",
        "voxel_nm": [8, 8, 8],
        "organelles_segmented": ["mito", "er", "nucleus"],
        "s3_path": f"s3://{DATASETS_BUCKET}/jrc_macrophage-2/",
    },
    {
        "id": "jrc_mus-liver",
        "title": "Mouse liver",
        "organism": "Mus musculus",
        "cell_type": "hepatocyte",
        "imaging": "FIB-SEM",
        "voxel_nm": [8, 8, 8],
        "organelles_segmented": ["mito", "er", "nucleus"],
        "s3_path": f"s3://{DATASETS_BUCKET}/jrc_mus-liver/",
    },
    {
        "id": "jrc_fly-fsb-1",
        "title": "Drosophila Fan-Shaped Body",
        "organism": "Drosophila melanogaster",
        "cell_type": "neuron",
        "imaging": "FIB-SEM",
        "voxel_nm": [8, 8, 8],
        "organelles_segmented": ["mito", "er", "nucleus"],
        "s3_path": f"s3://{DATASETS_BUCKET}/jrc_fly-fsb-1/",
    },
]

TIMEOUT = 30.0


@mcp.tool()
async def list_datasets(organism: str = "", cell_type: str = "") -> str:
    """List known OpenOrganelle datasets, optionally filtered.

    Note: OpenOrganelle has no API, so this uses a curated catalog.
    The full list may not include all datasets on S3.

    Args:
        organism: Filter by organism (e.g., "Homo sapiens"). Case-insensitive.
        cell_type: Filter by cell type (e.g., "HeLa"). Case-insensitive.
    """
    results = KNOWN_DATASETS

    if organism:
        organism_lower = organism.lower()
        results = [d for d in results if organism_lower in d["organism"].lower()]

    if cell_type:
        cell_type_lower = cell_type.lower()
        results = [d for d in results if cell_type_lower in d["cell_type"].lower()]

    return json.dumps(
        {
            "total": len(results),
            "note": "This is a curated subset. Use list_s3_bucket to discover all datasets.",
            "datasets": results,
        },
        indent=2,
    )


@mcp.tool()
async def get_dataset_metadata(dataset_id: str) -> str:
    """Get metadata for a specific OpenOrganelle dataset by reading its attributes.json from S3.

    Args:
        dataset_id: Dataset ID (e.g., "jrc_hela-2")
    """
    # Try to fetch attributes.json from S3 via HTTPS
    s3_url = f"https://{DATASETS_BUCKET}.s3.amazonaws.com/{dataset_id}/attributes.json"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(s3_url)
        if resp.status_code == 200:
            attrs = resp.json()
            return json.dumps(
                {
                    "dataset_id": dataset_id,
                    "attributes": attrs,
                    "s3_path": f"s3://{DATASETS_BUCKET}/{dataset_id}/",
                    "https_url": f"https://{DATASETS_BUCKET}.s3.amazonaws.com/{dataset_id}/",
                    "python_example": (
                        f"import fibsem_tools as fst\n"
                        f"data = fst.read_xarray('s3://{DATASETS_BUCKET}/{dataset_id}/em/fibsem-uint8/s0', "
                        f"storage_options={{'anon': True}})"
                    ),
                },
                indent=2,
            )
        else:
            # Fall back to known catalog
            match = [d for d in KNOWN_DATASETS if d["id"] == dataset_id]
            if match:
                return json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "from_catalog": True,
                        "metadata": match[0],
                        "note": "Could not fetch attributes.json from S3. Showing catalog data.",
                    },
                    indent=2,
                )
            return json.dumps(
                {"error": f"Dataset '{dataset_id}' not found in catalog or S3."}
            )


@mcp.tool()
async def get_access_info(dataset_id: str) -> str:
    """Get data access URLs and Python code snippets for an OpenOrganelle dataset.

    Args:
        dataset_id: Dataset ID (e.g., "jrc_hela-2")
    """
    return json.dumps(
        {
            "dataset_id": dataset_id,
            "s3_path": f"s3://{DATASETS_BUCKET}/{dataset_id}/",
            "neuroglancer_url": f"https://openorganelle.janelia.org/datasets/{dataset_id}",
            "aws_cli": f"aws s3 ls --no-sign-request s3://{DATASETS_BUCKET}/{dataset_id}/",
            "python_fibsem_tools": (
                f"import fibsem_tools as fst\n"
                f"# Load EM data as lazy dask-backed xarray\n"
                f"em = fst.read_xarray(\n"
                f"    's3://{DATASETS_BUCKET}/{dataset_id}/em/fibsem-uint8/s0',\n"
                f"    storage_options={{'anon': True}}\n"
                f")\n"
                f"print(em)  # shows dimensions, coords, chunk sizes"
            ),
            "python_zarr": (
                f"import zarr\n"
                f"import s3fs\n"
                f"fs = s3fs.S3FileSystem(anon=True)\n"
                f"store = s3fs.S3Map(root='{DATASETS_BUCKET}/{dataset_id}/', s3=fs)\n"
                f"z = zarr.open(store, mode='r')\n"
                f"print(list(z.keys()))"
            ),
            "formats": {
                "raw_em": "N5 or Zarr (multiscale, chunked)",
                "segmentations": "N5 or Zarr label volumes",
                "web_viewer": "Neuroglancer Precomputed",
            },
            "license": "CC-BY-4.0",
        },
        indent=2,
    )


@mcp.tool()
async def list_s3_bucket(prefix: str = "", max_keys: int = 100) -> str:
    """List contents of the OpenOrganelle S3 bucket to discover datasets.

    Uses the S3 REST API (ListObjectsV2) to enumerate datasets without
    needing boto3 installed.

    Args:
        prefix: S3 key prefix to list under (e.g., "jrc_hela-2/" for a specific dataset)
        max_keys: Max keys to return
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        params = {
            "list-type": "2",
            "delimiter": "/",
            "max-keys": str(max_keys),
        }
        if prefix:
            params["prefix"] = prefix

        resp = await client.get(
            f"https://{DATASETS_BUCKET}.s3.amazonaws.com/",
            params=params,
        )
        resp.raise_for_status()

        # Parse XML response
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        prefixes = [
            cp.find("s3:Prefix", ns).text
            for cp in root.findall("s3:CommonPrefixes", ns)
            if cp.find("s3:Prefix", ns) is not None
        ]

        objects = []
        for content in root.findall("s3:Contents", ns):
            key = content.find("s3:Key", ns)
            size = content.find("s3:Size", ns)
            if key is not None:
                objects.append(
                    {
                        "key": key.text,
                        "size_bytes": int(size.text) if size is not None else 0,
                    }
                )

        return json.dumps(
            {
                "bucket": DATASETS_BUCKET,
                "prefix": prefix or "(root)",
                "directories": prefixes,
                "files": objects[:20],  # Truncate for readability
                "total_files": len(objects),
            },
            indent=2,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
