"""Scanner for Cell Image Library (CIL).

CIL is a public resource of peer-reviewed cell biology images, videos, and
animations. Covers a wide range of organisms and imaging modalities.
"""

from __future__ import annotations

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

CIL_API = "http://www.cellimagelibrary.org"


class CellImageLibraryScanner(BaseScanner):
    """Scan Cell Image Library for microscopy datasets."""

    name = "CellImageLibrary"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        queries = [
            ("fluorescence confocal", "fluorescence"),
            ("light sheet", "fluorescence"),
            ("electron microscopy", "em"),
            ("immunofluorescence", "fluorescence"),
        ]

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            seen_ids: set[str] = set()
            for query, modality_class in queries:
                try:
                    resp = await client.get(
                        f"{CIL_API}/images",
                        params={"search": query, "format": "json"},
                        follow_redirects=True,
                    )
                    if resp.status_code != 200:
                        continue

                    try:
                        data = resp.json()
                    except Exception:
                        continue

                    items = data if isinstance(data, list) else data.get("images", data.get("results", []))
                    if not isinstance(items, list):
                        continue

                    for item in items[:limit]:
                        if not isinstance(item, dict):
                            continue
                        cil_id = str(item.get("CIL_CCDB", {}).get("CIL_ID", item.get("id", "")))
                        if not cil_id or cil_id in seen_ids:
                            continue
                        seen_ids.add(cil_id)

                        organism = ""
                        if isinstance(item.get("CIL_CCDB", {}).get("NCBI_ORGANISM", {}), dict):
                            organism = item["CIL_CCDB"]["NCBI_ORGANISM"].get("ORGANISM_COMMON", "")

                        title = item.get("CIL_CCDB", {}).get("CIL", {}).get("CORE", {}).get("TERMSANDCONDITIONS", {}).get("free_text", "")
                        if not title:
                            title = f"CIL image {cil_id}"

                        imaging = item.get("CIL_CCDB", {}).get("CIL", {}).get("CORE", {}).get("IMAGINGMODE", [])
                        if isinstance(imaging, list):
                            imaging = ", ".join(str(m.get("free_text", m)) if isinstance(m, dict) else str(m) for m in imaging)

                        results.append(DiscoveredDataset(
                            id=f"cil_{cil_id}",
                            repository="CellImageLibrary",
                            title=title[:120],
                            organism=organism,
                            imaging_modality=imaging[:80] if isinstance(imaging, str) else "",
                            data_format="tiff",
                            access_url=f"{CIL_API}/images/{cil_id}",
                            provenance=f"Cell Image Library search for '{query}'",
                            modality_class=modality_class,
                            supports_random_access=False,
                        ))
                except Exception as e:
                    print(f"  [{self.name}] API error for '{query}': {e}")

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
