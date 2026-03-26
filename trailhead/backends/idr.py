"""Backend for IDR (Image Data Resource) datasets.

IDR provides some datasets as OME-Zarr on EBI's S3 endpoint. This backend
reads OME-Zarr arrays using zarr + s3fs.

Most IDR data is light microscopy, not EM. Some datasets include label
images alongside raw, which can serve as segmentations.
"""

from __future__ import annotations

import json
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
        self._voxel_cache: dict[str, tuple[float, float, float] | None] = {}

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

    def _read_ome_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float] | None:
        """Read voxel sizes from OME-Zarr .zattrs multiscales metadata."""
        cache_key = f"{entry.id}:{scale}"
        if cache_key in self._voxel_cache:
            return self._voxel_cache[cache_key]

        raw_path = self._resolve_raw_path(entry)
        try:
            zattrs_path = f"{raw_path}/.zattrs"
            with self._fs.open(zattrs_path, "r") as f:
                attrs = json.load(f)
            multiscales = attrs.get("multiscales", [])
            if multiscales:
                ms = multiscales[0]
                axes = ms.get("axes", [])
                datasets = ms.get("datasets", [])
                idx = min(scale, len(datasets) - 1) if datasets else 0
                if idx < len(datasets):
                    transforms = datasets[idx].get("coordinateTransformations", [])
                    for t in transforms:
                        if t.get("type") == "scale":
                            s = t["scale"]
                            # OME-Zarr axes order varies; find spatial dims
                            axis_names = [a.get("name", "") for a in axes] if axes else []
                            if axis_names:
                                # Map axis name -> scale value
                                axis_map = dict(zip(axis_names, s))
                                z_val = axis_map.get("z", 0)
                                y_val = axis_map.get("y", 0)
                                x_val = axis_map.get("x", 0)
                                # OME-Zarr units may be micrometers
                                unit = next(
                                    (a.get("unit", "") for a in axes if a.get("name") == "x"),
                                    "",
                                )
                                factor = 1000.0 if unit in ("micrometer", "micrometre", "µm") else 1.0
                                if z_val > 0 and y_val > 0 and x_val > 0:
                                    result = (z_val * factor, y_val * factor, x_val * factor)
                                    self._voxel_cache[cache_key] = result
                                    return result
                            else:
                                # No axes info — assume last 3 are (z, y, x)
                                if len(s) >= 3:
                                    result = (float(s[-3]), float(s[-2]), float(s[-1]))
                                    self._voxel_cache[cache_key] = result
                                    return result
        except Exception:
            pass
        self._voxel_cache[cache_key] = None
        return None

    def get_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float]:
        vox = self._read_ome_voxel_size(entry, scale)
        if vox:
            return vox
        return super().get_voxel_size(entry, scale)

    def has_voxel_metadata(self, entry: DatasetEntry) -> bool:
        vox = self._read_ome_voxel_size(entry, 0)
        if vox is not None:
            return True
        return super().has_voxel_metadata(entry)

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
    ) -> NDArray:
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

        return np.asarray(data)

    def read_raw_crop_multichannel(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
        channels: list[int] | None = None,
    ) -> NDArray:
        """Read all channels from an OME-Zarr dataset. Returns (C, Z, Y, X)."""
        arr = self._open_array(self._resolve_raw_path(entry), scale)
        z, y, x = offset
        dz, dy, dx = shape
        ndim = len(arr.shape)

        if ndim == 5:  # (t, c, z, y, x)
            data = arr[0, :, z : z + dz, y : y + dy, x : x + dx]
        elif ndim == 4:  # (c, z, y, x)
            data = arr[:, z : z + dz, y : y + dy, x : x + dx]
        else:  # (z, y, x) — single channel
            data = arr[z : z + dz, y : y + dy, x : x + dx]
            data = np.asarray(data)[np.newaxis, ...]

        result = np.asarray(data)
        if channels is not None:
            result = result[channels]
        return result

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
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
