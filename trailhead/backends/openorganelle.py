"""Backend for OpenOrganelle (Janelia/CellMap) datasets on S3.

Data is stored as multiscale N5 arrays in public S3 buckets
(janelia-cosem-datasets) with anonymous access. Uses tensorstore
for efficient chunked reads.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import tensorstore as ts
from numpy.typing import NDArray

from trailhead.backends.base import Backend
from trailhead.registry import DatasetEntry

BUCKET = "janelia-cosem-datasets"


class OpenOrganelleBackend(Backend):
    """Read crops from OpenOrganelle N5 volumes on S3."""

    @lru_cache(maxsize=32)
    def _open_array(self, s3_path: str, scale: int) -> ts.TensorStore:
        """Open an N5 array at the given scale level."""
        spec = {
            "driver": "n5",
            "kvstore": {
                "driver": "s3",
                "bucket": BUCKET,
                "path": f"{s3_path}/s{scale}",
            },
        }
        return ts.open(spec, read=True).result()

    def _resolve_raw_path(self, entry: DatasetEntry) -> str:
        if entry.raw_path:
            return entry.raw_path
        return f"{entry.id}/{entry.id}.n5/em/fibsem-uint16"

    def _resolve_seg_path(self, entry: DatasetEntry, organelle: str) -> str:
        if organelle in entry.segmentation_paths:
            return entry.segmentation_paths[organelle]
        return f"{entry.id}/{entry.id}.n5/labels/{organelle}_seg"

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        arr = self._open_array(self._resolve_raw_path(entry), scale)
        return tuple(arr.shape)

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        arr = self._open_array(self._resolve_raw_path(entry), scale)
        z, y, x = offset
        dz, dy, dx = shape
        data = arr[z : z + dz, y : y + dy, x : x + dx].read().result()
        return np.asarray(data, dtype=np.uint8)

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        seg_path = self._resolve_seg_path(entry, organelle)
        arr = self._open_array(seg_path, scale)
        z, y, x = offset
        dz, dy, dx = shape
        data = arr[z : z + dz, y : y + dy, x : x + dx].read().result()
        return np.asarray(data, dtype=np.uint8)
