"""Scanner for Allen Institute datasets.

Covers Allen Cell Explorer (3D fluorescence cell structure) with actual S3
data paths resolved from the allencell bucket, and Allen Brain Observatory
(catalog-only, NWB format not yet supported).
"""

from __future__ import annotations

import httpx
import s3fs

from micro_agent.discover import DiscoveredDataset
from micro_agent.scanners.base import BaseScanner

ALLEN_BRAIN_API = "https://api.brain-map.org/api/v2/data/query.json"
ALLEN_BUCKET = "allencell"

# Subdirectory names in Allen Cell packages that typically contain raw 3D TIFFs
_RAW_SUBDIRS = [
    "crop_raw", "cell_images_3d", "crop_seg", "crop_raw_channels",
]

_TIFF_SUFFIXES = (".ome.tif", ".ome.tiff", ".tiff", ".tif")


def _is_tiff(path: str) -> bool:
    return path.lower().endswith(_TIFF_SUFFIXES)


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

        The bucket structure is typically:
            allencell/aics/{dataset_name}/{subfolder}/{file}.ome.tif
        where subfolder is e.g. crop_raw, cell_images_3d, etc.
        We use fs.glob() to search up to 4 levels deep for TIFFs.
        """
        try:
            fs = s3fs.S3FileSystem(anon=True)

            # List top-level dataset packages
            try:
                top_dirs = fs.ls(f"{ALLEN_BUCKET}/aics/")
            except Exception:
                top_dirs = []

            if not top_dirs:
                try:
                    top_dirs = fs.ls(ALLEN_BUCKET)
                    top_dirs = [d for d in top_dirs if not d.endswith((".md", ".txt", ".json"))]
                except Exception as e:
                    print(f"  [{self.name}] S3 bucket listing failed: {e}")
                    return

            resolved = 0
            for dir_path in sorted(top_dirs):
                if resolved >= limit:
                    break
                pkg_name = dir_path.split("/")[-1]
                if not pkg_name or pkg_name.startswith("."):
                    continue

                ome_tiff_path = None
                try:
                    ome_tiff_path = self._find_tiff(fs, dir_path)
                except Exception:
                    pass

                if ome_tiff_path:
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
                        raw_path=f"s3://{ome_tiff_path}",
                        provenance="Allen Cell S3 bucket scan with resolved path",
                        modality_class="fluorescence",
                        num_channels=4,
                        channel_names=["Brightfield", "DNA", "Cell membrane", "Structure"],
                        supports_random_access=True,
                    ))
                # Skip unresolved packages — don't add catalog-only entries
                # that can't be loaded

            print(f"  [{self.name}] Resolved {resolved}/{len(top_dirs)} Cell Explorer packages")

        except Exception as e:
            print(f"  [{self.name}] Cell Explorer error: {e}")

    @staticmethod
    def _find_tiff(fs: "s3fs.S3FileSystem", dir_path: str) -> str | None:
        """Find the first TIFF in a dataset package, searching up to 3 levels deep.

        Strategy:
        1. Check known raw-data subdirectory names first (fast path)
        2. Fall back to glob for any TIFF up to 3 levels deep
        """
        # Fast path: check known subdirectory names that contain raw 3D images
        for subdir in _RAW_SUBDIRS:
            sub_path = f"{dir_path}/{subdir}"
            try:
                if fs.exists(sub_path):
                    files = fs.ls(sub_path)
                    for f in files:
                        if _is_tiff(f):
                            return f
            except Exception:
                continue

        # Check for TIFFs directly in the package directory
        try:
            contents = fs.ls(dir_path)
            for item in contents:
                if not fs.isdir(item) and _is_tiff(item):
                    return item
        except Exception:
            pass

        # Broad search: glob for any TIFF up to 3 levels deep
        for pattern in [f"{dir_path}/*/*.tif", f"{dir_path}/*/*/*.tif",
                        f"{dir_path}/*/*.ome.tif", f"{dir_path}/*/*/*.ome.tif"]:
            try:
                matches = fs.glob(pattern)
                if matches:
                    return matches[0]
            except Exception:
                continue

        return None

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
