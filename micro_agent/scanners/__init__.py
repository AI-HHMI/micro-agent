"""Dataset scanners for each microscopy data source."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from micro_agent.discover import DiscoveredDataset
from micro_agent.scanners.base import BaseScanner
from micro_agent.scanners.openorganelle import OpenOrganelleScanner
from micro_agent.scanners.empiar import EMPIARScanner
from micro_agent.scanners.idr import IDRScanner
from micro_agent.scanners.bia import BioImageArchiveScanner
from micro_agent.scanners.allen import AllenScanner
from micro_agent.scanners.hpa import HPAScanner
from micro_agent.scanners.cell_image_lib import CellImageLibraryScanner
from micro_agent.scanners.zenodo import ZenodoScanner
from micro_agent.scanners.huggingface import HuggingFaceScanner
from micro_agent.scanners.openalex import OpenAlexScanner
from micro_agent.scanners.bossdb import BossDBScanner

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

    import time
    from concurrent.futures import ThreadPoolExecutor

    # Scanners that use blocking s3fs calls must run in threads
    # so they don't starve the async event loop.
    _BLOCKING_SCANNERS = {"Allen", "OpenOrganelle"}

    loop = asyncio.get_event_loop()
    thread_pool = ThreadPoolExecutor(max_workers=4)

    async def _timed_scan(scanner: BaseScanner) -> list[DiscoveredDataset]:
        t0 = time.time()
        try:
            if scanner.name in _BLOCKING_SCANNERS:
                # Run blocking scanner in a thread to free the event loop
                result = await loop.run_in_executor(
                    thread_pool,
                    lambda s=scanner: asyncio.run(s.scan(limit=limit)),
                )
            else:
                result = await scanner.scan(limit=limit)
            elapsed = time.time() - t0
            print(f"  [{scanner.name}] Found {len(result)} datasets ({elapsed:.1f}s)")
            return result
        except Exception as e:
            print(f"  [{scanner.name}] Scanner failed: {e} ({time.time() - t0:.1f}s)")
            return []

    async def _heartbeat(start: float, scanner_names: list[str]) -> None:
        """Print elapsed time every 60s so you can tell it's still running."""
        while True:
            await asyncio.sleep(60)
            elapsed = time.time() - start
            pending = [n for n in scanner_names if n not in completed]
            if pending:
                print(f"  ... {elapsed:.0f}s elapsed, still waiting on: {', '.join(pending)}")

    completed: set[str] = set()

    async def _timed_scan_tracked(scanner: BaseScanner) -> list[DiscoveredDataset]:
        result = await _timed_scan(scanner)
        completed.add(scanner.name)
        return result

    t_start = time.time()
    heartbeat_task = asyncio.create_task(_heartbeat(t_start, [s.name for s in scanners]))
    try:
        tasks = [_timed_scan_tracked(scanner) for scanner in scanners]
        all_results = await asyncio.gather(*tasks)
    finally:
        heartbeat_task.cancel()
        thread_pool.shutdown(wait=False)

    combined: list[DiscoveredDataset] = []
    for result in all_results:
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
