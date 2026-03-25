"""Backend for OpenOrganelle (Janelia/CellMap) datasets on S3.

Data is stored as both Zarr and N5 arrays in the public S3 bucket
(janelia-cosem-datasets). Zarr is preferred (already uint8, standard z,y,x layout).
Falls back to N5 if Zarr is unavailable.

IMPORTANT: Zarr arrays are stored as (z, y, x) but N5 arrays are stored as
(x, y, z). This backend normalizes everything to (z, y, x).
"""

from __future__ import annotations

from functools import lru_cache

import httpx
import numpy as np
import tensorstore as ts
from numpy.typing import NDArray

from trailhead.backends.base import Backend
from trailhead.registry import DatasetEntry

BUCKET = "janelia-cosem-datasets"
BUCKET_URL = f"https://{BUCKET}.s3.amazonaws.com"


class OpenOrganelleBackend(Backend):
    """Read crops from OpenOrganelle volumes on S3 (Zarr preferred, N5 fallback)."""

    def __init__(self) -> None:
        # Track which path/driver was actually opened for display and axis handling
        self._resolved_raw_paths: dict[str, str] = {}
        self._resolved_seg_paths: dict[str, str] = {}
        self._drivers: dict[str, str] = {}  # path -> "zarr" or "n5"
        self._voxel_size_cache: dict[str, tuple[float, float, float]] = {}

    def get_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float]:
        cache_key = entry.id
        if cache_key not in self._voxel_size_cache:
            base = self._read_base_voxel_size(entry)
            self._voxel_size_cache[cache_key] = base
        base = self._voxel_size_cache[cache_key]
        factor = 2 ** scale
        return (base[0] * factor, base[1] * factor, base[2] * factor)

    def _read_base_voxel_size(self, entry: DatasetEntry) -> tuple[float, float, float]:
        """Read base voxel size from N5 or zarr attributes on S3."""
        ds_id = entry.id
        # Try zarr .zattrs (OME multiscales) first
        zarr_zattrs_url = f"{BUCKET_URL}/{ds_id}/{ds_id}.zarr/recon-1/em/fibsem-uint8/.zattrs"
        try:
            resp = httpx.get(zarr_zattrs_url, timeout=15)
            if resp.status_code == 200:
                attrs = resp.json()
                if "multiscales" in attrs:
                    ms = attrs["multiscales"][0]
                    datasets = ms.get("datasets", [])
                    if datasets:
                        transforms = datasets[0].get("coordinateTransformations", [])
                        for t in transforms:
                            if t.get("type") == "scale":
                                s = t["scale"]
                                # OME zarr scale is in axes order (z, y, x)
                                return (float(s[0]), float(s[1]), float(s[2]))
        except Exception:
            pass

        # Try N5 s0/attributes.json
        if entry.raw_path:
            n5_attrs_url = f"{BUCKET_URL}/{entry.raw_path}/s0/attributes.json"
        else:
            n5_attrs_url = f"{BUCKET_URL}/{ds_id}/{ds_id}.n5/em/fibsem-uint16/s0/attributes.json"
        try:
            resp = httpx.get(n5_attrs_url, timeout=15)
            if resp.status_code == 200:
                attrs = resp.json()
                if "pixelResolution" in attrs:
                    dims = attrs["pixelResolution"]["dimensions"]
                    # N5 pixelResolution follows data axes order [x, y, z]
                    return (float(dims[2]), float(dims[1]), float(dims[0]))
        except Exception:
            pass

        # Fallback to registry or default
        if entry.voxel_size_nm and len(entry.voxel_size_nm) >= 3:
            return (entry.voxel_size_nm[0], entry.voxel_size_nm[1], entry.voxel_size_nm[2])
        return (8.0, 8.0, 8.0)

    @lru_cache(maxsize=32)
    def _open_array(self, s3_path: str, scale: int, driver: str = "zarr") -> ts.TensorStore:
        """Open an array at the given scale level."""
        spec = {
            "driver": driver,
            "kvstore": {
                "driver": "s3",
                "bucket": BUCKET,
                "path": f"{s3_path}/s{scale}",
            },
            "context": {
                "aws_credentials": {"type": "anonymous"},
            },
        }
        return ts.open(spec, read=True).result()

    def _open_with_fallback(self, zarr_path: str, n5_path: str, scale: int) -> tuple[ts.TensorStore, str, str]:
        """Try zarr first, fall back to N5. Returns (store, actual_path, driver)."""
        try:
            arr = self._open_array(zarr_path, scale, driver="zarr")
            return arr, zarr_path, "zarr"
        except Exception:
            arr = self._open_array(n5_path, scale, driver="n5")
            return arr, n5_path, "n5"

    def _resolve_paths(self, entry: DatasetEntry) -> tuple[str, str]:
        """Return (zarr_path, n5_path) for raw EM data."""
        ds_id = entry.id
        zarr_path = f"{ds_id}/{ds_id}.zarr/recon-1/em/fibsem-uint8"
        if entry.raw_path:
            n5_path = entry.raw_path
        else:
            n5_path = f"{ds_id}/{ds_id}.n5/em/fibsem-uint16"
        return zarr_path, n5_path

    def _resolve_seg_paths(self, entry: DatasetEntry, organelle: str) -> tuple[str, str]:
        """Return (zarr_path, n5_path) for segmentation data."""
        ds_id = entry.id
        zarr_path = f"{ds_id}/{ds_id}.zarr/recon-1/labels/{organelle}_seg"
        if organelle in entry.segmentation_paths:
            n5_path = entry.segmentation_paths[organelle]
        else:
            n5_path = f"{ds_id}/{ds_id}.n5/labels/{organelle}_seg"
        return zarr_path, n5_path

    def _read_crop(
        self, arr: ts.TensorStore, driver: str,
        offset: tuple[int, int, int], shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Read a crop, handling axis order differences between zarr and N5.

        Zarr is (z, y, x) — read directly.
        N5 is (x, y, z) — reverse indices and transpose result.
        """
        z, y, x = offset
        dz, dy, dx = shape

        if driver == "n5":
            # N5 stores as (x, y, z) — swap offset/shape accordingly
            data = arr[x : x + dx, y : y + dy, z : z + dz].read().result()
            return np.asarray(np.transpose(data, (2, 1, 0)))
        else:
            # Zarr stores as (z, y, x) — read directly
            data = arr[z : z + dz, y : y + dy, x : x + dx].read().result()
            return np.asarray(data)

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        zarr_path, n5_path = self._resolve_paths(entry)
        arr, actual, driver = self._open_with_fallback(zarr_path, n5_path, scale)
        self._resolved_raw_paths[entry.id] = actual
        self._drivers[actual] = driver
        shape = tuple(arr.shape)
        if driver == "n5":
            # N5 shape is (x, y, z) — return as (z, y, x)
            return (shape[2], shape[1], shape[0])
        return shape

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        zarr_path, n5_path = self._resolve_paths(entry)
        arr, actual, driver = self._open_with_fallback(zarr_path, n5_path, scale)
        self._resolved_raw_paths[entry.id] = actual
        self._drivers[actual] = driver

        data = self._read_crop(arr, driver, offset, shape)

        # Normalize to uint8 if needed (zarr is already uint8, N5 may be uint16)
        if data.dtype != np.uint8:
            dmin, dmax = float(data.min()), float(data.max())
            if dmax > dmin:
                data = ((data.astype(np.float32) - dmin) / (dmax - dmin) * 255).astype(np.uint8)
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
        zarr_path, n5_path = self._resolve_seg_paths(entry, organelle)
        arr, actual, driver = self._open_with_fallback(zarr_path, n5_path, scale)
        self._resolved_seg_paths[f"{entry.id}/{organelle}"] = actual
        self._drivers[actual] = driver

        data = self._read_crop(arr, driver, offset, shape)
        return np.asarray(data, dtype=np.uint8)

    def get_resolved_raw_path(self, entry_id: str) -> str:
        """Return the actual S3 path used for raw data (zarr or n5)."""
        return self._resolved_raw_paths.get(entry_id, "")

    def get_resolved_seg_path(self, entry_id: str, organelle: str) -> str:
        """Return the actual S3 path used for segmentation data."""
        return self._resolved_seg_paths.get(f"{entry_id}/{organelle}", "")
