"""Backend for Neuroglancer Precomputed volumes on GCS and S3.

Handles datasets from MICrONS, FlyEM, Open NeuroData, and other sources
that store EM + segmentation as Neuroglancer Precomputed format.
Uses tensorstore for efficient chunked reads with anonymous access.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
import numpy as np
import tensorstore as ts
from numpy.typing import NDArray

from trailhead.backends.base import Backend
from trailhead.registry import DatasetEntry


class MICrONSBackend(Backend):
    """Read crops from Neuroglancer Precomputed volumes (GCS or S3)."""

    def __init__(self) -> None:
        self._info_cache: dict[str, dict] = {}

    def _fetch_info(self, url: str) -> dict:
        """Fetch and cache the neuroglancer precomputed info JSON."""
        if url in self._info_cache:
            return self._info_cache[url]

        if url.startswith("gs://"):
            bucket, _, path = url[5:].partition("/")
            http_url = f"https://storage.googleapis.com/{bucket}/{path}/info"
        elif url.startswith("s3://"):
            bucket, _, path = url[5:].partition("/")
            http_url = f"https://{bucket}.s3.amazonaws.com/{path}/info"
        else:
            http_url = url.rstrip("/") + "/info"

        resp = httpx.get(http_url, timeout=30)
        resp.raise_for_status()
        info = resp.json()
        self._info_cache[url] = info
        return info

    def get_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float]:
        url = self._resolve_raw_url(entry)
        try:
            info = self._fetch_info(url)
            scales = info["scales"]
            idx = min(scale, len(scales) - 1)
            res = scales[idx]["resolution"]
            # neuroglancer precomputed resolution is [x, y, z] in nm
            return (float(res[2]), float(res[1]), float(res[0]))
        except Exception:
            return super().get_voxel_size(entry, scale)

    def get_num_scales(self, entry: DatasetEntry) -> int:
        url = self._resolve_raw_url(entry)
        try:
            info = self._fetch_info(url)
            return len(info["scales"])
        except Exception:
            return 6

    @lru_cache(maxsize=16)
    def _open_array(self, url: str, scale: int) -> ts.TensorStore:
        """Open a Neuroglancer precomputed volume at the given scale."""
        spec: dict = {
            "driver": "neuroglancer_precomputed",
            "kvstore": url,
            "scale_index": scale,
        }
        # Set anonymous credentials for S3-backed datasets to suppress IMDS noise
        if url.startswith("s3://"):
            spec["context"] = {"aws_credentials": {"type": "anonymous"}}
        return ts.open(spec, read=True).result()

    def _resolve_raw_url(self, entry: DatasetEntry) -> str:
        if entry.raw_path:
            return entry.raw_path
        return "gs://iarpa_microns/minnie/minnie65/em"

    def _resolve_seg_url(self, entry: DatasetEntry, organelle: str) -> str:
        if organelle in entry.segmentation_paths:
            return entry.segmentation_paths[organelle]
        return "gs://iarpa_microns/minnie/minnie65/seg"

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        arr = self._open_array(self._resolve_raw_url(entry), scale)
        # Neuroglancer precomputed domain is (x, y, z, channel)
        # Return as (z, y, x) shape
        return (arr.shape[2], arr.shape[1], arr.shape[0])

    def _read_crop(
        self, arr: ts.TensorStore, offset: tuple[int, int, int], shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Read a crop, translating 0-based (z,y,x) offset to the volume's actual domain."""
        z, y, x = offset
        dz, dy, dx = shape

        # Get the volume origin (x_min, y_min, z_min, channel_min)
        origin = arr.domain.origin
        x_off, y_off, z_off = int(origin[0]), int(origin[1]), int(origin[2])

        # Translate to absolute coordinates
        data = arr[
            x_off + x : x_off + x + dx,
            y_off + y : y_off + y + dy,
            z_off + z : z_off + z + dz,
            0,
        ].read().result()

        # Transpose from (x, y, z) to (z, y, x)
        return np.transpose(np.asarray(data), (2, 1, 0))

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        arr = self._open_array(self._resolve_raw_url(entry), scale)
        data = self._read_crop(arr, offset, shape)
        return np.asarray(data, dtype=np.uint8)

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        seg_url = self._resolve_seg_url(entry, organelle)
        arr = self._open_array(seg_url, scale)
        data = self._read_crop(arr, offset, shape)
        # Convert uint64 instance labels to uint8 binary mask
        return np.asarray(data > 0, dtype=np.uint8) * 255
