"""Scanner for BossDB (Brain Observatory Storage Service & Database).

BossDB is a public cloud archive for petascale volumetric neuroimaging data,
hosted on AWS by JHU/APL. Data is stored in Neuroglancer Precomputed format
on S3 (s3://bossdb-open-data/). No auth required for public datasets.

Docs: https://bossdb.org
S3: s3://bossdb-open-data/
"""

from __future__ import annotations

import httpx
import s3fs

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

BOSSDB_S3_BUCKET = "bossdb-open-data"
BOSSDB_S3_URL = f"https://{BOSSDB_S3_BUCKET}.s3.amazonaws.com"

# Top-level prefixes that are infrastructure, not datasets
_SKIP_PREFIXES = {
    "_w-mirror", "graphs", "inventory", "mesh", "metadata",
    "query_results", "raw", "s3bak", "scratch",
}


class BossDBScanner(BaseScanner):
    """Scan BossDB S3 bucket for public neuroimaging datasets."""

    name = "BossDB"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        try:
            fs = s3fs.S3FileSystem(anon=True)
            top_dirs = fs.ls(BOSSDB_S3_BUCKET)

            for dir_path in sorted(top_dirs):
                if len(results) >= limit:
                    break

                collection = dir_path.split("/")[-1]
                if not collection or collection in _SKIP_PREFIXES:
                    continue

                # Each collection can have multiple experiments
                try:
                    experiments = fs.ls(dir_path)
                except Exception:
                    continue

                for exp_path in experiments:
                    if len(results) >= limit:
                        break

                    exp_name = exp_path.split("/")[-1]
                    if not exp_name or exp_name.startswith("."):
                        continue

                    # Look for channels with precomputed info files
                    try:
                        channels = fs.ls(exp_path)
                    except Exception:
                        continue

                    raw_channel = None
                    seg_channel = None
                    all_channels = []

                    for ch_path in channels:
                        ch_name = ch_path.split("/")[-1]
                        if not ch_name:
                            continue

                        # Check for info file (precomputed format marker)
                        info_path = f"{ch_path}/info"
                        try:
                            if not fs.exists(info_path):
                                continue
                        except Exception:
                            continue

                        all_channels.append(ch_name)

                        # Classify channel
                        ch_lower = ch_name.lower()
                        if any(kw in ch_lower for kw in ["em", "image", "raw", "grayscale"]):
                            if raw_channel is None:
                                raw_channel = ch_name
                        elif any(kw in ch_lower for kw in ["seg", "label", "annotation", "mask"]):
                            if seg_channel is None:
                                seg_channel = ch_name

                    if not all_channels:
                        continue

                    # Default to first channel if no obvious raw channel
                    if raw_channel is None:
                        raw_channel = all_channels[0]

                    precomputed_url = (
                        f"precomputed://https://{BOSSDB_S3_BUCKET}.s3.amazonaws.com"
                        f"/{collection}/{exp_name}/{raw_channel}"
                    )

                    # Read voxel size from info file if possible
                    voxel_size = []
                    try:
                        import json
                        info_data = json.loads(
                            fs.cat_file(f"{BOSSDB_S3_BUCKET}/{collection}/{exp_name}/{raw_channel}/info")
                        )
                        scales = info_data.get("scales", [])
                        if scales:
                            res = scales[0].get("resolution", [])
                            if len(res) >= 3:
                                # Precomputed resolution is in nm (x, y, z)
                                voxel_size = [float(res[2]), float(res[1]), float(res[0])]
                    except Exception:
                        pass

                    seg_paths = {}
                    if seg_channel:
                        seg_url = (
                            f"precomputed://https://{BOSSDB_S3_BUCKET}.s3.amazonaws.com"
                            f"/{collection}/{exp_name}/{seg_channel}"
                        )
                        seg_paths["segmentation"] = seg_url

                    ds_id = f"bossdb_{collection}_{exp_name}"
                    results.append(DiscoveredDataset(
                        id=ds_id,
                        repository="BossDB",
                        title=f"BossDB — {collection}/{exp_name}",
                        organism="",
                        imaging_modality="electron microscopy",
                        has_raw=True,
                        has_segmentation=bool(seg_channel),
                        data_format="neuroglancer-precomputed",
                        access_url=precomputed_url,
                        raw_path=precomputed_url,
                        segmentation_paths=seg_paths,
                        voxel_size_nm=voxel_size,
                        provenance="BossDB S3 bucket scan",
                        modality_class="em",
                        supports_random_access=True,
                    ))

        except Exception as e:
            print(f"  [{self.name}] Error scanning S3 bucket: {e}")

        return results[:limit]

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        """Verify the precomputed info file is accessible via HTTPS."""
        if not dataset.raw_path:
            return "pending"
        try:
            # Extract HTTPS URL from precomputed:// URI
            https_url = dataset.raw_path.replace("precomputed://", "")
            info_url = f"{https_url}/info"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(info_url)
                return "verified" if resp.status_code == 200 else "failed"
        except Exception:
            return "failed"
