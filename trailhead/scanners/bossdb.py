"""Scanner for BossDB (Brain Observatory Storage Service & Database).

BossDB is a public cloud archive for petascale volumetric neuroimaging data,
hosted on AWS by JHU/APL. Data is stored in Neuroglancer Precomputed format
on S3 (s3://bossdb-open-data/). No auth required for public datasets.

Docs: https://bossdb.org
S3: s3://bossdb-open-data/
"""

from __future__ import annotations

from collections import defaultdict

import httpx
import s3fs

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

BOSSDB_S3_BUCKET = "bossdb-open-data"

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

            # Single glob to find all precomputed info files at once
            # Pattern: bucket/collection/experiment/channel/info
            info_files = fs.glob(f"{BOSSDB_S3_BUCKET}/*/*/*/info")

            # Group by (collection, experiment) → list of channel names
            experiments: dict[tuple[str, str], list[str]] = defaultdict(list)
            for path in info_files:
                # path: "bossdb-open-data/collection/experiment/channel/info"
                parts = path.split("/")
                if len(parts) < 5:
                    continue
                collection, exp_name, ch_name = parts[1], parts[2], parts[3]
                if collection in _SKIP_PREFIXES:
                    continue
                experiments[(collection, exp_name)].append(ch_name)

            for (collection, exp_name), channels in sorted(experiments.items()):
                if len(results) >= limit:
                    break

                # Classify channels
                raw_channel = None
                seg_channel = None
                for ch in channels:
                    ch_lower = ch.lower()
                    if any(kw in ch_lower for kw in ["em", "image", "raw", "grayscale"]):
                        if raw_channel is None:
                            raw_channel = ch
                    elif any(kw in ch_lower for kw in ["seg", "label", "annotation", "mask"]):
                        if seg_channel is None:
                            seg_channel = ch

                if raw_channel is None:
                    raw_channel = channels[0]

                precomputed_url = (
                    f"precomputed://https://{BOSSDB_S3_BUCKET}.s3.amazonaws.com"
                    f"/{collection}/{exp_name}/{raw_channel}"
                )

                seg_paths = {}
                if seg_channel:
                    seg_paths["segmentation"] = (
                        f"precomputed://https://{BOSSDB_S3_BUCKET}.s3.amazonaws.com"
                        f"/{collection}/{exp_name}/{seg_channel}"
                    )

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
            https_url = dataset.raw_path.replace("precomputed://", "")
            info_url = f"{https_url}/info"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(info_url)
                return "verified" if resp.status_code == 200 else "failed"
        except Exception:
            return "failed"
