"""Scanner for EMPIAR (Electron Microscopy Public Image Archive).

EMPIAR doesn't have a list-all-entries endpoint. Instead we query
individual known entry ranges and the latest citations to discover entries.
"""

from __future__ import annotations

import httpx

from micro_agent.discover import DiscoveredDataset
from micro_agent.scanners.base import BaseScanner

EMPIAR_API = "https://www.ebi.ac.uk/empiar/api"


class EMPIARScanner(BaseScanner):
    """Scan EMPIAR entries via the REST API."""

    name = "EMPIAR"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Strategy 1: Get recent entries via latest citations
            cited_ids: list[str] = []
            try:
                resp = await client.get(f"{EMPIAR_API}/latest_citations/")
                if resp.status_code == 200:
                    citations = resp.json()
                    if isinstance(citations, list):
                        for cite in citations:
                            pmid = cite.get("pmid", "")
                            if pmid:
                                cited_ids.append(pmid)
            except Exception as e:
                print(f"  [{self.name}] Citations error: {e}")

            # Strategy 2: Probe a range of known EMPIAR IDs
            # EMPIAR IDs are typically 5-digit numbers (10001-11xxx as of 2024)
            probe_ids = [str(i) for i in range(11000, 11000 - limit, -1)]

            # Query individual entries (EMPIAR only supports single-entry lookup)
            entry_ids_to_check = probe_ids[:limit]
            for entry_id in entry_ids_to_check:
                try:
                    resp = await client.get(f"{EMPIAR_API}/entry/{entry_id}/")
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    entry_key = f"EMPIAR-{entry_id}"
                    entry_data = data.get(entry_key, {})
                    if not entry_data:
                        continue

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

                    results.append(DiscoveredDataset(
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
                    ))

                    if len(results) >= limit:
                        break
                except Exception:
                    continue

        return results

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
