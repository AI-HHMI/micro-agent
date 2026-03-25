"""Backend for IDR (Image Data Resource) datasets.

IDR provides some datasets as OME-Zarr on EBI's S3 endpoint. This backend
reads OME-Zarr arrays using zarr + s3fs.

Most IDR data is light microscopy, not EM. Some datasets include label
images alongside raw, which can serve as segmentations.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import zarr
import s3fs
from numpy.typing import NDArray

from trailhead.backends.base import Backend
from trailhead.registry import DatasetEntry

EBI_S3_ENDPOINT = "https://uk1s3.embassy.ebi.ac.uk"
IDR_ZARR_PREFIX = "idr/zarr/v0.5"


class IDRBackend(Backend):
    """Read crops from IDR OME-Zarr datasets on EBI S3."""

    def __init__(self) -> None:
        self._fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={"endpoint_url": EBI_S3_ENDPOINT},
        )

    @lru_cache(maxsize=32)
    def _open_array(self, s3_path: str, scale: int) -> zarr.Array:
        """Open an OME-Zarr array at the given resolution level."""
        store = s3fs.S3Map(root=f"{s3_path}/{scale}", s3=self._fs)
        return zarr.open(store, mode="r")

    def _resolve_raw_path(self, entry: DatasetEntry) -> str:
        if entry.raw_path:
            return entry.raw_path
        # Default OME-Zarr path for IDR images
        return f"{IDR_ZARR_PREFIX}/{entry.id}.zarr"

    def _resolve_seg_path(self, entry: DatasetEntry, organelle: str) -> str:
        if organelle in entry.segmentation_paths:
            return entry.segmentation_paths[organelle]
        # IDR label images are typically in /labels/ within the OME-Zarr
        return f"{IDR_ZARR_PREFIX}/{entry.id}.zarr/labels/{organelle}"

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        arr = self._open_array(self._resolve_raw_path(entry), scale)
        shape = arr.shape
        # OME-Zarr may have (t, c, z, y, x) or (z, y, x) or (c, z, y, x)
        # Return last 3 dims as (z, y, x)
        if len(shape) >= 3:
            return shape[-3:]
        return shape

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
        ndim = len(arr.shape)

        if ndim == 5:  # (t, c, z, y, x)
            data = arr[0, 0, z : z + dz, y : y + dy, x : x + dx]
        elif ndim == 4:  # (c, z, y, x)
            data = arr[0, z : z + dz, y : y + dy, x : x + dx]
        else:  # (z, y, x)
            data = arr[z : z + dz, y : y + dy, x : x + dx]

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
        ndim = len(arr.shape)

        if ndim == 5:
            data = arr[0, 0, z : z + dz, y : y + dy, x : x + dx]
        elif ndim == 4:
            data = arr[0, z : z + dz, y : y + dy, x : x + dx]
        else:
            data = arr[z : z + dz, y : y + dy, x : x + dx]

        return np.asarray(data, dtype=np.uint8)
