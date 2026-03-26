"""Scanner for Zenodo microscopy datasets.

Zenodo is a general-purpose research data repository. Many microscopy
datasets are deposited there, especially as supplements to publications.
No API key required for public records.
"""

from __future__ import annotations

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

ZENODO_API = "https://zenodo.org/api/records/"

# File extensions that indicate microscopy image data
IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".czi", ".lif", ".nd2",
    ".zarr", ".n5",
    ".mrc", ".rec",
}


class ZenodoScanner(BaseScanner):
    """Scan Zenodo for public microscopy datasets."""

    name = "Zenodo"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        queries = [
            ("fluorescence microscopy dataset", "fluorescence"),
            ("confocal microscopy dataset", "fluorescence"),
            ("light sheet microscopy dataset", "fluorescence"),
            ("electron microscopy segmentation dataset", "em"),
            ("ome-tiff microscopy", "fluorescence"),
        ]

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            seen_ids: set[str] = set()
            for query, modality_class in queries:
                try:
                    resp = await client.get(
                        ZENODO_API,
                        params={
                            "q": query,
                            "type": "dataset",
                            "size": min(limit, 25),
                            "sort": "mostrecent",
                            "status": "published",
                        },
                    )
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])

                    for record in hits:
                        record_id = str(record.get("id", ""))
                        if not record_id or record_id in seen_ids:
                            continue
                        seen_ids.add(record_id)

                        metadata = record.get("metadata", {})
                        title = metadata.get("title", "")
                        description = metadata.get("description", "").lower()

                        # Check if record contains image files
                        files = record.get("files", [])
                        has_images = any(
                            any(f.get("key", "").lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
                            for f in files
                        )

                        # Determine data format from files
                        data_format = ""
                        for f in files:
                            key = f.get("key", "").lower()
                            if ".ome.tif" in key:
                                data_format = "ome-tiff"
                                break
                            elif key.endswith(".czi"):
                                data_format = "czi"
                                break
                            elif key.endswith(".lif"):
                                data_format = "lif"
                                break
                            elif key.endswith(".nd2"):
                                data_format = "nd2"
                                break
                            elif key.endswith((".tif", ".tiff")):
                                data_format = "tiff"
                            elif key.endswith(".zarr"):
                                data_format = "zarr"

                        # Extract organism from keywords
                        organism = ""
                        keywords = metadata.get("keywords", [])
                        for kw in keywords:
                            kw_lower = kw.lower()
                            if any(org in kw_lower for org in [
                                "homo sapiens", "mus musculus", "drosophila",
                                "c. elegans", "zebrafish", "arabidopsis",
                            ]):
                                organism = kw
                                break

                        doi = metadata.get("doi", "")
                        results.append(DiscoveredDataset(
                            id=f"zenodo_{record_id}",
                            repository="Zenodo",
                            title=title[:120] if title else f"Zenodo {record_id}",
                            organism=organism,
                            imaging_modality=query.split(" dataset")[0],
                            has_raw=has_images,
                            data_format=data_format,
                            access_url=f"https://zenodo.org/records/{record_id}",
                            provenance=f"Zenodo API search for '{query}'",
                            modality_class=modality_class,
                            supports_random_access=False,
                        ))
                except Exception as e:
                    print(f"  [{self.name}] API error for '{query}': {e}")

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
