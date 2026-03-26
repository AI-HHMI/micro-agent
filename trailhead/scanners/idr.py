"""Scanner for IDR (Image Data Resource) — both EM and fluorescence datasets.

Resolves actual OME-Zarr S3 paths on EBI S3. IDR stores plate-level zarr
files under zarr/v0.4/{study_prefix}/{plate_id}.zarr, where the study
prefix is derived from the screen name (e.g., idr0001-graml-sysgro/screenA
→ idr0001A).
"""

from __future__ import annotations

import asyncio
import re

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

IDR_API = "https://idr.openmicroscopy.org/api/v0/m"
IDR_WEBCLIENT_API = "https://idr.openmicroscopy.org/webclient/api"
IDR_S3_BASE = "https://uk1s3.embassy.ebi.ac.uk"
IDR_ZARR_PREFIX = "idr/zarr/v0.4"


def _study_prefix(screen_name: str) -> str | None:
    """Extract study prefix from screen name.

    Examples:
        'idr0001-graml-sysgro/screenA' → 'idr0001A'
        'idr0044-prelich-cell/screenA' → 'idr0044A'
        'idr0056-tsang-astraea/screenB' → 'idr0056B'
    """
    m = re.match(r"(idr\d+)-.*/screen([A-Z])", screen_name, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None


class IDRScanner(BaseScanner):
    """Scan IDR for OME-Zarr datasets, resolving actual S3 data paths."""

    name = "IDR"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Scan screens — resolve plate-level zarr paths in parallel
            try:
                resp = await client.get(
                    f"{IDR_API}/screens/",
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 200:
                    screens = resp.json().get("data", [])[:limit]

                    # Resolve all screen zarr paths concurrently
                    sem = asyncio.Semaphore(10)

                    async def _resolve(screen: dict) -> DiscoveredDataset | None:
                        async with sem:
                            screen_id = screen.get("@id") or screen.get("id", "")
                            name = screen.get("Name") or screen.get("name", "")
                            if not screen_id:
                                return None

                            desc = (screen.get("Description") or screen.get("description", "")).lower()
                            is_fluor = any(
                                kw in desc or kw in name.lower()
                                for kw in ["fluorescence", "confocal", "light sheet",
                                            "widefield", "gfp", "dapi", "fitc"]
                            )

                            raw_path = await self._resolve_screen_zarr(client, screen_id, name)
                            return DiscoveredDataset(
                                id=f"idr_screen_{screen_id}",
                                repository="IDR",
                                title=name[:120] if name else str(screen_id),
                                data_format="ome-zarr",
                                imaging_modality="fluorescence" if is_fluor else "OME-Zarr",
                                access_url=(
                                    f"https://idr.openmicroscopy.org/webclient/"
                                    f"?show=screen-{screen_id}"
                                ),
                                raw_path=raw_path or "",
                                provenance="IDR screens API" + (
                                    " with resolved plate zarr" if raw_path else " (catalog only)"
                                ),
                                modality_class="fluorescence" if is_fluor else "",
                                supports_random_access=bool(raw_path),
                            )

                    resolved = await asyncio.gather(*[_resolve(s) for s in screens])
                    results.extend(r for r in resolved if r is not None)
            except Exception as e:
                print(f"  [{self.name}] Screens API error: {e}")

            # Scan projects — these use dataset→image hierarchy, not plates
            try:
                resp = await client.get(
                    f"{IDR_API}/projects/",
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 200:
                    projects = resp.json().get("data", [])
                    for proj in projects[:limit]:
                        proj_id = proj.get("@id") or proj.get("id", "")
                        name = proj.get("Name") or proj.get("name", "")
                        if not proj_id:
                            continue

                        desc = (proj.get("Description") or proj.get("description", "")).lower()
                        is_fluor = any(
                            kw in desc or kw in name.lower()
                            for kw in ["fluorescence", "confocal", "light sheet",
                                        "widefield", "gfp", "dapi"]
                        )

                        # Projects don't map to plate zarrs; catalog-only for now
                        results.append(DiscoveredDataset(
                            id=f"idr_project_{proj_id}",
                            repository="IDR",
                            title=name[:120] if name else str(proj_id),
                            data_format="ome-zarr",
                            imaging_modality="fluorescence" if is_fluor else "",
                            access_url=(
                                f"https://idr.openmicroscopy.org/webclient/"
                                f"?show=project-{proj_id}"
                            ),
                            provenance="IDR projects API (catalog only)",
                            modality_class="fluorescence" if is_fluor else "",
                            supports_random_access=False,
                        ))
            except Exception as e:
                print(f"  [{self.name}] Projects API error: {e}")

        return results

    async def _resolve_screen_zarr(
        self, client: httpx.AsyncClient, screen_id: int | str, screen_name: str,
    ) -> str | None:
        """Resolve screen → study prefix → plate zarr → first well/field image path.

        IDR plate zarr structure: {plate_id}.zarr/{row}/{col}/{field}/
        Each field directory contains an OME-Zarr image with multiscale levels 0,1,2,...
        """
        try:
            prefix = _study_prefix(screen_name)
            if not prefix:
                return None

            # Get plates for this screen
            resp = await client.get(
                f"{IDR_WEBCLIENT_API}/plates/",
                params={"id": screen_id},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return None
            plates = resp.json().get("plates", [])
            if not plates:
                return None

            plate_id = plates[0].get("id")
            if not plate_id:
                return None

            # Read plate .zattrs to find first well path
            zarr_base = f"{IDR_ZARR_PREFIX}/{prefix}/{plate_id}.zarr"
            plate_url = f"{IDR_S3_BASE}/{zarr_base}/.zattrs"
            resp = await client.get(plate_url, timeout=8.0)
            if resp.status_code >= 400:
                return None

            plate_attrs = resp.json()
            wells = plate_attrs.get("plate", {}).get("wells", [])
            if not wells:
                return None

            well_path = wells[0].get("path")
            if not well_path:
                return None

            # Read well .zattrs to find first image/field
            well_url = f"{IDR_S3_BASE}/{zarr_base}/{well_path}/.zattrs"
            resp = await client.get(well_url, timeout=8.0)
            if resp.status_code >= 400:
                return None

            well_attrs = resp.json()
            images = well_attrs.get("well", {}).get("images", [])
            if not images:
                return None

            field_path = images[0].get("path", "0")

            # Full path to the image-level zarr (contains multiscale levels)
            return f"{zarr_base}/{well_path}/{field_path}"

        except Exception:
            return None

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        if not dataset.raw_path:
            return "pending"
        try:
            url = f"{IDR_S3_BASE}/{dataset.raw_path}/.zattrs"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(url)
                return "verified" if resp.status_code < 400 else "failed"
        except Exception:
            return "failed"
