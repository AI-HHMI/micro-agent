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
        """Scan Allen Cell Explorer S3 bucket for 3D fluorescence datasets."""
        try:
            fs = s3fs.S3FileSystem(anon=True)

            # List top-level dataset packages in the allencell bucket
            try:
                top_dirs = fs.ls(f"{ALLEN_BUCKET}/aics/")
            except Exception:
                top_dirs = []

            if not top_dirs:
                # Fallback: try listing the bucket root
                try:
                    top_dirs = fs.ls(ALLEN_BUCKET)
                    top_dirs = [d for d in top_dirs if not d.endswith((".md", ".txt", ".json"))]
                except Exception as e:
                    print(f"  [{self.name}] S3 bucket listing failed: {e}")
                    return

            for dir_path in sorted(top_dirs)[:limit]:
                pkg_name = dir_path.split("/")[-1]
                if not pkg_name or pkg_name.startswith("."):
                    continue

                # Look for OME-TIFF files in this package
                ome_tiff_path = None
                try:
                    # Check for common file patterns
                    contents = fs.ls(dir_path)
                    for item in contents[:20]:
                        name = item.split("/")[-1].lower()
                        if name.endswith((".ome.tif", ".ome.tiff", ".tiff", ".tif")):
                            ome_tiff_path = item
                            break
                    # If no direct TIFFs, look one level deeper
                    if not ome_tiff_path:
                        for subdir in contents[:5]:
                            if fs.isdir(subdir):
                                sub_contents = fs.ls(subdir)
                                for item in sub_contents[:10]:
                                    name = item.split("/")[-1].lower()
                                    if name.endswith((".ome.tif", ".ome.tiff", ".tiff", ".tif")):
                                        ome_tiff_path = item
                                        break
                                if ome_tiff_path:
                                    break
                except Exception:
                    pass

                if ome_tiff_path:
                    results.append(DiscoveredDataset(
                        id=f"allen_cell_{pkg_name}",
                        repository="Allen",
                        title=f"Allen Cell — {pkg_name}",
                        organism="Homo sapiens",
                        cell_type="hiPSC-derived",
                        imaging_modality="spinning disk confocal",
                        data_format="ome-tiff",
                        access_url=f"s3://{ALLEN_BUCKET}/",
                        raw_path=f"s3://{ome_tiff_path}",
                        provenance="Allen Cell S3 bucket scan with resolved path",
                        modality_class="fluorescence",
                        num_channels=4,
                        channel_names=["Brightfield", "DNA", "Cell membrane", "Structure"],
                        supports_random_access=True,
                    ))
                else:
                    results.append(DiscoveredDataset(
                        id=f"allen_cell_{pkg_name}",
                        repository="Allen",
                        title=f"Allen Cell — {pkg_name}",
                        organism="Homo sapiens",
                        cell_type="hiPSC-derived",
                        imaging_modality="spinning disk confocal",
                        data_format="ome-tiff",
                        access_url=f"s3://{ALLEN_BUCKET}/{dir_path}/",
                        provenance="Allen Cell S3 bucket (no TIFF resolved)",
                        modality_class="fluorescence",
                        supports_random_access=False,
                    ))

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
