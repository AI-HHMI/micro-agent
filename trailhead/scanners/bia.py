"""Scanner for BioImage Archive."""

from __future__ import annotations

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner


class BioImageArchiveScanner(BaseScanner):
    """Scan BioImage Archive for microscopy datasets via BioStudies API."""

    name = "BioImage Archive"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        queries = [
            ("electron microscopy segmentation", "em"),
            ("fluorescence microscopy", "fluorescence"),
            ("confocal microscopy", "fluorescence"),
            ("light sheet microscopy", "fluorescence"),
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            seen_ids: set[str] = set()
            for query, modality_class in queries:
                try:
                    resp = await client.get(
                        "https://www.ebi.ac.uk/biostudies/api/v1/search",
                        params={"query": query, "pageSize": limit},
                    )
                    if resp.status_code != 200:
                        continue
                    hits = resp.json().get("hits", [])
                    for hit in hits:
                        accession = hit.get("accession", "")
                        if not accession or accession in seen_ids:
                            continue
                        seen_ids.add(accession)
                        results.append(DiscoveredDataset(
                            id=accession,
                            repository="BioImage Archive",
                            title=hit.get("title", ""),
                            access_url=(
                                f"https://www.ebi.ac.uk/biostudies/studies/{accession}"
                            ),
                            provenance=f"BioStudies API search for '{query}'",
                            modality_class=modality_class,
                            supports_random_access=False,
                        ))
                except Exception as e:
                    print(f"  [{self.name}] API error for '{query}': {e}")

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(dataset.access_url)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
