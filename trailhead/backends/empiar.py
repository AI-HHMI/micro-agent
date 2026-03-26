"""Backend for EMPIAR (Electron Microscopy Public Image Archive).

EMPIAR stores raw EM data as individual image files (TIFF, MRC, etc.) served
over HTTPS. This backend downloads only the slices needed for a crop, caching
them locally to avoid re-downloading.

Most EMPIAR entries do NOT have paired segmentations.
"""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from trailhead.backends.base import Backend
from trailhead.registry import DatasetEntry

EMPIAR_FTP_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability"

# Local cache for downloaded slices
CACHE_DIR = Path.home() / ".cache" / "trailhead" / "empiar"

# Max total dataset size we'll attempt to work with (per-slice download is fine,
# but we won't index datasets with thousands of huge files)
MAX_DATASET_SIZE_GB = 50


class EMPIARBackend(Backend):
    """Read crops from EMPIAR by downloading individual TIFF/MRC slices on demand."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=60, follow_redirects=True)

    @lru_cache(maxsize=16)
    def _list_slices(self, entry_id: str, data_subdir: str) -> list[str]:
        """List and sort slice filenames from an EMPIAR entry's data directory.

        Returns filenames sorted by z-index (natural sort on numeric parts).
        """
        empiar_num = entry_id.replace("EMPIAR-", "")
        url = f"{EMPIAR_FTP_BASE}/{empiar_num}/data/{data_subdir}/"
        resp = self._client.get(url)
        resp.raise_for_status()

        # Parse href links for image files
        files = re.findall(
            r'href="([^"]+\.(?:tif|tiff|mrc|png|jpg))"', resp.text, re.IGNORECASE
        )

        # Natural sort by numeric parts to get correct z-order
        def sort_key(name: str) -> list:
            parts = re.split(r"(\d+)", name)
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        return sorted(files, key=sort_key)

    def _resolve_data_subdir(self, entry: DatasetEntry) -> str:
        """Get the data subdirectory path for an entry.

        Stored in raw_path field of the DatasetEntry, e.g.
        '20180813_platynereis_parapodia/raw_16bit'.
        """
        if entry.raw_path:
            return entry.raw_path
        # Try to auto-discover by listing the data dir
        empiar_num = entry.id.replace("EMPIAR-", "")
        url = f"{EMPIAR_FTP_BASE}/{empiar_num}/data/"
        resp = self._client.get(url)
        resp.raise_for_status()
        dirs = re.findall(r'href="([^"]+/)"', resp.text)
        # Filter out parent dir link
        dirs = [d.strip("/") for d in dirs if not d.startswith("/")]
        if not dirs:
            raise ValueError(f"No data directories found for {entry.id}")
        # If there's only one dir, use it; otherwise pick the first
        return dirs[0]

    def _download_slice(self, entry_id: str, data_subdir: str, filename: str) -> NDArray:
        """Download a single slice and return as numpy array. Uses local cache."""
        empiar_num = entry_id.replace("EMPIAR-", "")
        cache_path = CACHE_DIR / empiar_num / data_subdir / (filename + ".npy")

        if cache_path.exists():
            return np.load(cache_path)

        url = f"{EMPIAR_FTP_BASE}/{empiar_num}/data/{data_subdir}/{filename}"
        resp = self._client.get(url)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content))
        arr = np.asarray(img)

        # Cache to disk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)

        return arr

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        data_subdir = self._resolve_data_subdir(entry)
        slices = self._list_slices(entry.id, data_subdir)
        if not slices:
            raise ValueError(f"No image slices found for {entry.id}")

        # Read first slice to get (y, x) dimensions
        first = self._download_slice(entry.id, data_subdir, slices[0])
        nz = len(slices)
        ny, nx = first.shape[:2]

        # Downscale for scale > 0
        factor = 2 ** scale
        return (nz // factor, ny // factor, nx // factor)

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        data_subdir = self._resolve_data_subdir(entry)
        slices = self._list_slices(entry.id, data_subdir)

        z, y, x = offset
        dz, dy, dx = shape
        factor = 2 ** scale

        # Map back to full-res z indices
        z_start = z * factor
        z_end = z_start + dz * factor

        volume = []
        for zi in range(z_start, min(z_end, len(slices)), factor):
            arr = self._download_slice(entry.id, data_subdir, slices[zi])
            # Crop (y, x) region at full res, then downsample
            y_start, x_start = y * factor, x * factor
            crop = arr[y_start : y_start + dy * factor, x_start : x_start + dx * factor]
            if factor > 1:
                crop = crop[::factor, ::factor]
            volume.append(crop)

        data = np.stack(volume, axis=0)

        # Normalize to uint8
        if data.dtype != np.uint8:
            dmin, dmax = float(data.min()), float(data.max())
            if dmax > dmin:
                data = ((data.astype(np.float32) - dmin) / (dmax - dmin) * 255).astype(
                    np.uint8
                )
            else:
                data = np.zeros_like(data, dtype=np.uint8)
        return data

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        raise NotImplementedError(
            f"EMPIAR entry {entry.id}: No segmentation data available. "
            "Most EMPIAR entries contain raw data only."
        )
