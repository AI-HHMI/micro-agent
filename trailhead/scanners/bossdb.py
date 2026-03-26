"""Scanner for BossDB (Brain Observatory Storage Service & Database).

BossDB is a public cloud archive for petascale volumetric neuroimaging data,
hosted on AWS by JHU/APL. Data is stored in Neuroglancer Precomputed format
on S3 (s3://bossdb-open-data/). No auth required for public datasets.

Uses the BossDB REST API for fast discovery, then verifies which
datasets have precomputed data on S3 via HTTP HEAD checks.

Docs: https://bossdb.org
API: https://api.bossdb.io/v1/
S3: s3://bossdb-open-data/
"""

from __future__ import annotations

import asyncio

import httpx

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner

BOSSDB_API = "https://api.bossdb.io/v1"
BOSSDB_S3_BUCKET = "bossdb-open-data"
BOSSDB_S3_URL = f"https://{BOSSDB_S3_BUCKET}.s3.amazonaws.com"
_AUTH_HEADER = {"Authorization": "Token public"}

_CONCURRENT = 10  # Max parallel HTTP checks


class BossDBScanner(BaseScanner):
    """Scan BossDB via REST API for public neuroimaging datasets."""

    name = "BossDB"

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        results: list[DiscoveredDataset] = []

        async with httpx.AsyncClient(timeout=15.0, headers=_AUTH_HEADER) as client:
            # Step 1: List all collections
            try:
                resp = await client.get(f"{BOSSDB_API}/collection/")
                resp.raise_for_status()
                collections = resp.json().get("collections", [])
            except Exception as e:
                print(f"  [{self.name}] Failed to list collections: {e}")
                return []

            # Step 2: Get experiments for each collection concurrently
            sem = asyncio.Semaphore(_CONCURRENT)

            async def _get_experiments(col: str) -> list[str]:
                async with sem:
                    try:
                        r = await client.get(f"{BOSSDB_API}/collection/{col}/experiment/")
                        if r.status_code == 200:
                            return r.json().get("experiments", [])
                    except Exception:
                        pass
                    return []

            exp_lists = await asyncio.gather(*[_get_experiments(c) for c in collections])

            # Step 3: Get channels and verify S3 precomputed data concurrently
            async def _process_experiment(col: str, exp: str) -> DiscoveredDataset | None:
                async with sem:
                    try:
                        # Get channels
                        r = await client.get(
                            f"{BOSSDB_API}/collection/{col}/experiment/{exp}/channel/"
                        )
                        if r.status_code != 200:
                            return None
                        channels = r.json().get("channels", [])
                        if not channels:
                            return None

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

                        # Verify precomputed data exists on S3
                        info_url = f"{BOSSDB_S3_URL}/{col}/{exp}/{raw_channel}/info"
                        r = await client.head(info_url)
                        if r.status_code != 200:
                            return None

                        precomputed_url = (
                            f"precomputed://{BOSSDB_S3_URL}"
                            f"/{col}/{exp}/{raw_channel}"
                        )

                        seg_paths = {}
                        if seg_channel:
                            seg_paths["segmentation"] = (
                                f"precomputed://{BOSSDB_S3_URL}"
                                f"/{col}/{exp}/{seg_channel}"
                            )

                        return DiscoveredDataset(
                            id=f"bossdb_{col}_{exp}",
                            repository="BossDB",
                            title=f"BossDB — {col}/{exp}",
                            organism="",
                            imaging_modality="electron microscopy",
                            has_raw=True,
                            has_segmentation=bool(seg_channel),
                            data_format="neuroglancer-precomputed",
                            access_url=precomputed_url,
                            raw_path=precomputed_url,
                            segmentation_paths=seg_paths,
                            provenance="BossDB REST API + S3 verification",
                            modality_class="em",
                            supports_random_access=True,
                        )
                    except Exception:
                        return None

            # Build list of (collection, experiment) pairs
            tasks = []
            for col, exps in zip(collections, exp_lists):
                for exp in exps:
                    tasks.append(_process_experiment(col, exp))

            all_results = await asyncio.gather(*tasks)
            results = [r for r in all_results if r is not None]

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
