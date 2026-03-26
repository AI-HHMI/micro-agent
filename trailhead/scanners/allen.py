"""Scanner for Allen Institute datasets.

Covers Allen Cell Explorer (3D fluorescence cell structure) with actual S3
data paths resolved from the allencell bucket, and Allen Brain Observatory
(catalog-only, NWB format not yet supported).
"""

from __future__ import annotations

import httpx
import s3fs

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

ALLEN_BRAIN_API = "https://api.brain-map.org/api/v2/data/query.json"
ALLEN_BUCKET = "allencell"


class AllenScanner(BaseScanner):
    """Scan Allen Institute data portals for microscopy datasets."""

    name = "Allen"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        # Allen Cell Explorer — resolve actual S3 OME-TIFF paths
        await self._scan_cell_explorer(results, limit)

        # Allen Brain Observatory — catalog only (NWB format)
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._scan_brain_observatory(client, results, limit)

        return results[:limit]

    async def _scan_cell_explorer(
        self,
        results: list[DiscoveredDataset],
        limit: int,
    ) -> None:
        """Scan Allen Cell Explorer S3 bucket for 3D fluorescence datasets.

        Uses a single broad glob to find all TIFFs at once, then groups
        by dataset package name to avoid per-package S3 calls.
        """
        try:
            fs = s3fs.S3FileSystem(anon=True)

            # Single glob to find all TIFFs under aics/
            all_tiffs: list[str] = []
            for pattern in [f"{ALLEN_BUCKET}/aics/*/*.tif*",
                            f"{ALLEN_BUCKET}/aics/*/*/*.tif*",
                            f"{ALLEN_BUCKET}/aics/*/*/*/*.tif*"]:
                try:
                    all_tiffs.extend(fs.glob(pattern))
                except Exception:
                    continue

            # Group by package name (first directory after aics/)
            # path: "allencell/aics/{pkg_name}/..."
            pkg_tiffs: dict[str, str] = {}  # pkg_name → first tiff path
            for tiff_path in all_tiffs:
                parts = tiff_path.split("/")
                if len(parts) < 3:
                    continue
                # Find the index of "aics" and take the next part
                try:
                    aics_idx = parts.index("aics")
                    pkg_name = parts[aics_idx + 1]
                except (ValueError, IndexError):
                    continue
                if pkg_name and pkg_name not in pkg_tiffs:
                    pkg_tiffs[pkg_name] = tiff_path

            resolved = 0
            for pkg_name, tiff_path in sorted(pkg_tiffs.items()):
                if resolved >= limit:
                    break
                resolved += 1
                results.append(DiscoveredDataset(
                    id=f"allen_cell_{pkg_name}",
                    repository="Allen",
                    title=f"Allen Cell — {pkg_name}",
                    organism="Homo sapiens",
                    cell_type="hiPSC-derived",
                    imaging_modality="spinning disk confocal",
                    data_format="ome-tiff",
                    access_url=f"s3://{ALLEN_BUCKET}/",
                    raw_path=f"s3://{tiff_path}",
                    provenance="Allen Cell S3 bucket scan with resolved path",
                    modality_class="fluorescence",
                    num_channels=4,
                    channel_names=["Brightfield", "DNA", "Cell membrane", "Structure"],
                    supports_random_access=True,
                ))

            # Also count unresolved packages for logging
            try:
                top_dirs = fs.ls(f"{ALLEN_BUCKET}/aics/")
                total = len([d for d in top_dirs if not d.split("/")[-1].startswith(".")])
            except Exception:
                total = resolved
            print(f"  [{self.name}] Resolved {resolved}/{total} Cell Explorer packages")

        except Exception as e:
            print(f"  [{self.name}] Cell Explorer error: {e}")

    async def _scan_brain_observatory(
        self,
        client: httpx.AsyncClient,
        results: list[DiscoveredDataset],
        limit: int,
    ) -> None:
        """Allen Brain Observatory — catalog only (NWB format, not loadable yet)."""
        try:
            resp = await client.get(
                ALLEN_BRAIN_API,
                params={
                    "criteria": "model::WellKnownFile,rma::criteria,[well_known_file_type_id$eq486262171]",
                    "num_rows": min(limit, 20),
                    "start_row": 0,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    results.append(DiscoveredDataset(
                        id="allen_brain_observatory",
                        repository="Allen",
                        title="Allen Brain Observatory — Two-photon calcium imaging",
                        organism="Mus musculus",
                        cell_type="visual cortex neuron",
                        imaging_modality="two-photon",
                        data_format="nwb",
                        access_url="https://observatory.brain-map.org/visualcoding",
                        provenance="Allen Brain Map API",
                        modality_class="fluorescence",
                        supports_random_access=False,
                    ))
        except Exception as e:
            print(f"  [{self.name}] Brain Observatory error: {e}")

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        if not dataset.raw_path or not dataset.raw_path.startswith("s3://"):
            return "pending"
        try:
            fs = s3fs.S3FileSystem(anon=True)
            path = dataset.raw_path.replace("s3://", "")
            fs.info(path)
            return "verified"
        except Exception:
            return "failed"
