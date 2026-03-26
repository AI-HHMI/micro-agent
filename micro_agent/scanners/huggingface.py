"""Scanner for HuggingFace Datasets Hub.

HuggingFace hosts a growing number of microscopy and bioimage datasets,
including cell painting screens, fluorescence collections, and EM data.
No API key required for public datasets.

API docs: https://huggingface.co/docs/hub/api#get-apimodels
"""

from __future__ import annotations

import httpx

from micro_agent.discover import DiscoveredDataset
from micro_agent.scanners.base import BaseScanner

HF_API = "https://huggingface.co/api/datasets"

# Search queries targeting microscopy data on HuggingFace
QUERIES = [
    ("microscopy", ""),
    ("fluorescence cell", "fluorescence"),
    ("cell painting", "fluorescence"),
    ("electron microscopy segmentation", "em"),
    ("confocal", "fluorescence"),
    ("ome-tiff", ""),
    ("histopathology", "fluorescence"),
    ("light sheet", "fluorescence"),
]

# File extensions indicating microscopy image data
IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".czi",
    ".lif",
    ".nd2",
    ".zarr",
    ".n5",
    ".mrc",
    ".rec",
    ".png",
    ".jpg",  # common for 2D datasets
}


class HuggingFaceScanner(BaseScanner):
    """Scan HuggingFace Datasets Hub for public microscopy datasets."""

    name = "HuggingFace"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []
        seen_ids: set[str] = set()

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for query, modality_hint in QUERIES:
                if len(results) >= limit:
                    break
                try:
                    resp = await client.get(
                        HF_API,
                        params={
                            "search": query,
                            "sort": "downloads",
                            "direction": -1,
                            "limit": min(limit, 20),
                        },
                    )
                    if resp.status_code != 200:
                        continue

                    for ds in resp.json():
                        ds_id = ds.get("id", "")
                        if not ds_id or ds_id in seen_ids:
                            continue
                        seen_ids.add(ds_id)

                        tags = ds.get("tags", [])
                        description = ds.get("description", "") or ""

                        # Infer modality from tags/description
                        modality_class = modality_hint
                        if not modality_class:
                            desc_lower = description.lower()
                            tags_lower = " ".join(tags).lower()
                            combined = desc_lower + " " + tags_lower
                            if any(
                                kw in combined
                                for kw in [
                                    "fluoresc",
                                    "confocal",
                                    "light sheet",
                                    "cell painting",
                                ]
                            ):
                                modality_class = "fluorescence"
                            elif any(
                                kw in combined
                                for kw in ["electron microscopy", "em ", "sem ", "tem "]
                            ):
                                modality_class = "em"

                        # Extract organism from tags
                        organism = ""
                        for tag in tags:
                            tag_lower = tag.lower()
                            if any(
                                org in tag_lower
                                for org in [
                                    "homo sapiens",
                                    "human",
                                    "mus musculus",
                                    "mouse",
                                    "drosophila",
                                    "c. elegans",
                                    "zebrafish",
                                    "arabidopsis",
                                ]
                            ):
                                organism = tag
                                break

                        results.append(
                            DiscoveredDataset(
                                id=f"hf_{ds_id.replace('/', '_')}",
                                repository="HuggingFace",
                                title=ds_id,
                                organism=organism,
                                imaging_modality=query,
                                has_raw=True,
                                data_format="huggingface-dataset",
                                access_url=f"https://huggingface.co/datasets/{ds_id}",
                                provenance=f"HuggingFace API search for '{query}'",
                                modality_class=modality_class or "fluorescence",
                                supports_random_access=False,
                            )
                        )

                except Exception as e:
                    print(f"  [{self.name}] API error for '{query}': {e}")

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        """Check that the dataset page is accessible."""
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
