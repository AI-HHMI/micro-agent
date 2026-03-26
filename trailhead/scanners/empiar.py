"""Scanner for EMPIAR (Electron Microscopy Public Image Archive).

EMPIAR doesn't have a list-all-entries endpoint. Instead we query
individual known entry ranges and the latest citations to discover entries.
"""

from __future__ import annotations

import asyncio

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

EMPIAR_API = "https://www.ebi.ac.uk/empiar/api"

_CONCURRENT = 10  # Max parallel API requests


class EMPIARScanner(BaseScanner):
    """Scan EMPIAR entries via the REST API."""

    name = "EMPIAR"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Probe a range of known EMPIAR IDs concurrently
            probe_ids = [str(i) for i in range(11000, 11000 - limit, -1)]

            sem = asyncio.Semaphore(_CONCURRENT)

            async def _fetch_entry(entry_id: str) -> DiscoveredDataset | None:
                async with sem:
                    try:
                        resp = await client.get(f"{EMPIAR_API}/entry/{entry_id}/")
                        if resp.status_code != 200:
                            return None
                        data = resp.json()
                        entry_key = f"EMPIAR-{entry_id}"
                        entry_data = data.get(entry_key, {})
                        if not entry_data:
                            return None

                        title = entry_data.get("title", "")
                        organism = ""
                        org_data = entry_data.get("corresponding_author", {}).get("organism", "")
                        if isinstance(org_data, str):
                            organism = org_data

                        imagesets = entry_data.get("imagesets", [])
                        data_format = "mrc"
                        for imgset in imagesets:
                            fmt = imgset.get("data_format", "").lower()
                            if fmt:
                                data_format = fmt
                                break

                        return DiscoveredDataset(
                            id=entry_key,
                            repository="EMPIAR",
                            title=title[:120] if title else entry_key,
                            organism=organism,
                            imaging_modality="EM",
                            has_segmentation=False,
                            data_format=data_format,
                            access_url=f"https://www.ebi.ac.uk/empiar/entry/{entry_id}/",
                            provenance="EMPIAR single-entry API probe",
                            modality_class="em",
                            supports_random_access=False,
                        )
                    except Exception:
                        return None

            results = await asyncio.gather(*[_fetch_entry(eid) for eid in probe_ids])

        return [r for r in results if r is not None][:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
