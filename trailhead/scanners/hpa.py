"""Scanner for Human Protein Atlas (HPA) subcellular localization data.

HPA provides immunofluorescence images of proteins in human cells, with
subcellular localization annotations. The data is freely available via
their API and image server.
"""

from __future__ import annotations

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

HPA_API = "https://www.proteinatlas.org"


class HPAScanner(BaseScanner):
    """Scan Human Protein Atlas for immunofluorescence datasets."""

    name = "HPA"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Query HPA subcellular section for proteins with IF images
            try:
                resp = await client.get(
                    f"{HPA_API}/api/search_download.php",
                    params={
                        "search": "",
                        "format": "json",
                        "columns": "g,up,scl,if_images",
                        "compress": "no",
                    },
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else []
                    for item in data[:limit]:
                        gene = item.get("Gene", "")
                        uniprot = item.get("Uniprot", "")
                        locations = item.get("Subcellular location", "")
                        if not gene:
                            continue

                        # HPA API may return locations as list or semicolon-separated string
                        if isinstance(locations, list):
                            organelles = [loc.strip().lower() for loc in locations if loc.strip()]
                            locations_str = "; ".join(locations)
                        elif isinstance(locations, str) and locations:
                            organelles = [loc.strip().lower() for loc in locations.split(";") if loc.strip()]
                            locations_str = locations
                        else:
                            organelles = []
                            locations_str = ""

                        results.append(DiscoveredDataset(
                            id=f"hpa_{gene}_{uniprot}",
                            repository="HPA",
                            title=f"HPA — {gene} immunofluorescence ({locations_str})",
                            organism="Homo sapiens",
                            cell_type="various",
                            imaging_modality="confocal",
                            organelles=organelles,
                            has_segmentation=False,
                            data_format="tiff",
                            access_url=f"{HPA_API}/{uniprot}/subcellular",
                            provenance="HPA subcellular API",
                            modality_class="fluorescence",
                            num_channels=4,
                            channel_names=["DAPI", "Microtubules", "ER", gene],
                            wavelengths_nm=[405.0, 488.0, 561.0, 647.0],
                            fluorophores=["DAPI", "anti-tubulin", "anti-calreticulin", f"anti-{gene}"],
                            supports_random_access=False,
                        ))
            except Exception as e:
                print(f"  [{self.name}] API error: {e}")

        # If API didn't return data, add known HPA collections
        if not results:
            results.append(DiscoveredDataset(
                id="hpa_subcellular_collection",
                repository="HPA",
                title="Human Protein Atlas — Subcellular protein localization atlas",
                organism="Homo sapiens",
                cell_type="various",
                imaging_modality="confocal",
                data_format="tiff",
                access_url=f"{HPA_API}/humanproteome/subcellular",
                provenance="HPA known collection",
                modality_class="fluorescence",
                num_channels=4,
                channel_names=["DAPI", "Microtubules", "ER", "Protein"],
                supports_random_access=False,
            ))

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(dataset.access_url, follow_redirects=True)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
