"""Dataset scanners for each microscopy data source."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from trailhead.discover import DiscoveredDataset
from trailhead.scanners.base import BaseScanner
from trailhead.scanners.openorganelle import OpenOrganelleScanner
from trailhead.scanners.empiar import EMPIARScanner
from trailhead.scanners.idr import IDRScanner
from trailhead.scanners.bia import BioImageArchiveScanner
from trailhead.scanners.allen import AllenScanner
from trailhead.scanners.hpa import HPAScanner
from trailhead.scanners.cell_image_lib import CellImageLibraryScanner
from trailhead.scanners.zenodo import ZenodoScanner
from trailhead.scanners.huggingface import HuggingFaceScanner
from trailhead.scanners.openalex import OpenAlexScanner
from trailhead.scanners.bossdb import BossDBScanner

ALL_SCANNERS: list[type[BaseScanner]] = [
    OpenOrganelleScanner,
    EMPIARScanner,
    IDRScanner,
    BioImageArchiveScanner,
    AllenScanner,
    HPAScanner,
    CellImageLibraryScanner,
    ZenodoScanner,
    HuggingFaceScanner,
    OpenAlexScanner,
    # BossDBScanner — WIP, too slow for default discovery (see scanners/bossdb.py)
]

__all__ = [
    "BaseScanner",
    "OpenOrganelleScanner",
    "EMPIARScanner",
    "IDRScanner",
    "BioImageArchiveScanner",
    "AllenScanner",
    "HPAScanner",
    "CellImageLibraryScanner",
    "ZenodoScanner",
    "HuggingFaceScanner",
    "OpenAlexScanner",
    "BossDBScanner",
    "ALL_SCANNERS",
    "run_all_scanners",
]


async def _run_all_async(
    scanners: list[BaseScanner] | None = None,
    limit: int = 50,
    validate: bool = False,
) -> list[DiscoveredDataset]:
    """Run all scanners concurrently and return combined results."""
    if scanners is None:
        scanners = [cls() for cls in ALL_SCANNERS]

    tasks = [scanner.scan(limit=limit) for scanner in scanners]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[DiscoveredDataset] = []
    for scanner, result in zip(scanners, all_results):
        if isinstance(result, Exception):
            print(f"  [{scanner.name}] Scanner failed: {result}")
            continue
        print(f"  [{scanner.name}] Found {len(result)} datasets")
        combined.extend(result)

    if validate:
        print(f"\nValidating {len(combined)} datasets...")
        scanner_map = {s.name: s for s in scanners}
        for ds in combined:
            scanner = scanner_map.get(ds.repository)
            if scanner:
                ds.validation_status = await scanner.validate_access(ds)

    return combined


def run_all_scanners(
    scanners: list[BaseScanner] | None = None,
    limit: int = 50,
    validate: bool = False,
) -> list[DiscoveredDataset]:
    """Run all scanners and return combined results. Synchronous wrapper."""
    return asyncio.run(_run_all_async(scanners, limit, validate))
