"""Backend for OpenOrganelle (Janelia/CellMap) datasets on S3.

Data is stored as both Zarr and N5 arrays in the public S3 bucket
(janelia-cosem-datasets). Zarr is preferred (standard z,y,x layout).
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

from micro_agent.backends.base import Backend
from micro_agent.registry import DatasetEntry

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
        self._num_scales_cache: dict[str, int] = {}  # path -> num_scales

    def get_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float]:
        raw_path = entry.raw_path or f"{entry.id}/{entry.id}.n5/em/fibsem-uint16"
        num = self._read_num_scales(raw_path)
        if scale >= num:
            raise IndexError(f"Scale {scale} out of range (max {num - 1}) for {entry.id}")
        cache_key = entry.id
        if cache_key not in self._voxel_size_cache:
            base = self._read_base_voxel_size(entry)
            self._voxel_size_cache[cache_key] = base
        base = self._voxel_size_cache[cache_key]
        factor = 2 ** scale
        return (base[0] * factor, base[1] * factor, base[2] * factor)

    def _read_base_voxel_size(self, entry: DatasetEntry) -> tuple[float, float, float]:
        """Read base voxel size from zarr or N5 attributes on S3.

        Uses entry.raw_path to determine format rather than hardcoding paths.
        """
        bucket = self._bucket_for(entry)
        bucket_url = f"https://{bucket}.s3.amazonaws.com"
        raw_path = entry.raw_path or f"{entry.id}/{entry.id}.n5/em/fibsem-uint16"

        # If raw_path is zarr, try .zattrs first
        if ".zarr/" in raw_path:
            zattrs_url = f"{bucket_url}/{raw_path}/.zattrs"
            try:
                resp = httpx.get(zattrs_url, timeout=15)
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
                                    return (float(s[0]), float(s[1]), float(s[2]))
            except Exception:
                pass

        # Try N5 s0/attributes.json
        n5_attrs_url = f"{bucket_url}/{raw_path}/s0/attributes.json"
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
        if entry.voxel_size_nm and len(entry.voxel_size_nm) >= 3 and any(v > 0 for v in entry.voxel_size_nm[:3]):
            return (entry.voxel_size_nm[0], entry.voxel_size_nm[1], entry.voxel_size_nm[2])
        return self._DEFAULT_VOXEL_NM

    def _read_num_scales(self, path: str) -> int:
        """Probe how many scale levels exist for a given volume path."""
        if path in self._num_scales_cache:
            return self._num_scales_cache[path]

        bucket_url = f"https://{BUCKET}.s3.amazonaws.com"

        # For zarr, read from .zattrs multiscales datasets list
        if ".zarr/" in path:
            try:
                resp = httpx.get(f"{bucket_url}/{path}/.zattrs", timeout=15)
                if resp.status_code == 200:
                    attrs = resp.json()
                    ms = attrs.get("multiscales", [{}])[0]
                    n = len(ms.get("datasets", []))
                    if n > 0:
                        self._num_scales_cache[path] = n
                        return n
            except Exception:
                pass

        # For N5, probe s0, s1, ... until we get a 404
        for s in range(10):
            try:
                resp = httpx.get(f"{bucket_url}/{path}/s{s}/attributes.json", timeout=5)
                if resp.status_code != 200:
                    n = max(s, 1)
                    self._num_scales_cache[path] = n
                    return n
            except Exception:
                n = max(s, 1)
                self._num_scales_cache[path] = n
                return n

        self._num_scales_cache[path] = 6
        return 6

    def get_seg_voxel_size(
        self, entry: DatasetEntry, organelle: str, scale: int = 0,
    ) -> tuple[float, float, float]:
        seg_path, _ = self._resolve_seg_paths(entry, organelle)
        num = self._read_num_scales(seg_path)
        if scale >= num:
            raise IndexError(f"Seg scale {scale} out of range (max {num - 1}) for {entry.id}/{organelle}")
        cache_key = f"{entry.id}/{organelle}"
        if cache_key not in self._voxel_size_cache:
            base = self._read_seg_base_voxel_size(entry, organelle)
            self._voxel_size_cache[cache_key] = base
        base = self._voxel_size_cache[cache_key]
        factor = 2 ** scale
        return (base[0] * factor, base[1] * factor, base[2] * factor)

    def _read_seg_base_voxel_size(
        self, entry: DatasetEntry, organelle: str,
    ) -> tuple[float, float, float]:
        """Read base voxel size for a segmentation volume."""
        bucket = self._bucket_for(entry)
        bucket_url = f"https://{bucket}.s3.amazonaws.com"
        seg_path, _ = self._resolve_seg_paths(entry, organelle)

        if ".zarr/" in seg_path:
            zattrs_url = f"{bucket_url}/{seg_path}/.zattrs"
            try:
                resp = httpx.get(zattrs_url, timeout=15)
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
                                    return (float(s[0]), float(s[1]), float(s[2]))
            except Exception:
                pass

        n5_attrs_url = f"{bucket_url}/{seg_path}/s0/attributes.json"
        try:
            resp = httpx.get(n5_attrs_url, timeout=15)
            if resp.status_code == 200:
                attrs = resp.json()
                if "pixelResolution" in attrs:
                    dims = attrs["pixelResolution"]["dimensions"]
                    return (float(dims[2]), float(dims[1]), float(dims[0]))
        except Exception:
            pass

        # Fall back to raw voxel size
        return self._read_base_voxel_size(entry)


    def has_voxel_metadata(self, entry: DatasetEntry) -> bool:
        """OpenOrganelle reads voxel sizes from zarr/N5 attributes."""
        # Trigger the read so the cache is populated
        try:
            self.get_voxel_size(entry, 0)
            vox = self._voxel_size_cache.get(entry.id)
            return vox is not None and vox != self._DEFAULT_VOXEL_NM
        except Exception:
            return super().has_voxel_metadata(entry)

    def _bucket_for(self, entry: DatasetEntry) -> str:
        """Extract S3 bucket name from entry.access_url, default to BUCKET."""
        url = entry.access_url
        if url.startswith("s3://"):
            return url[5:].split("/")[0]
        return BUCKET

    @lru_cache(maxsize=32)
    def _open_array(self, s3_path: str, scale: int, driver: str = "zarr", bucket: str = BUCKET) -> ts.TensorStore:
        """Open an array at the given scale level."""
        spec = {
            "driver": driver,
            "kvstore": {
                "driver": "s3",
                "bucket": bucket,
                "path": f"{s3_path}/s{scale}",
            },
            "context": {
                "aws_credentials": {"type": "anonymous"},
            },
        }
        return ts.open(spec, read=True).result()

    def _resolve_paths(self, entry: DatasetEntry) -> tuple[str, str]:
        """Return (path, bucket) for raw EM data using the stored raw_path."""
        bucket = self._bucket_for(entry)
        if entry.raw_path:
            return entry.raw_path, bucket
        ds_id = entry.id
        return f"{ds_id}/{ds_id}.n5/em/fibsem-uint16", bucket

    def _resolve_seg_paths(self, entry: DatasetEntry, organelle: str) -> tuple[str, str]:
        """Return (path, bucket) for segmentation data using stored paths."""
        bucket = self._bucket_for(entry)
        if organelle in entry.segmentation_paths:
            return entry.segmentation_paths[organelle], bucket
        ds_id = entry.id
        return f"{ds_id}/{ds_id}.n5/labels/{organelle}_seg", bucket

    def _detect_driver(self, path: str) -> str:
        """Detect driver from path: zarr if path contains .zarr/, else n5."""
        return "zarr" if ".zarr/" in path else "n5"

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

    def _open_path(self, path: str, scale: int, bucket: str) -> tuple[ts.TensorStore, str]:
        """Open an array at the given path, returning (store, driver)."""
        driver = self._detect_driver(path)
        arr = self._open_array(path, scale, driver=driver, bucket=bucket)
        return arr, driver

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        path, bucket = self._resolve_paths(entry)
        arr, driver = self._open_path(path, scale, bucket)
        self._resolved_raw_paths[entry.id] = path
        self._drivers[path] = driver
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
        path, bucket = self._resolve_paths(entry)
        arr, driver = self._open_path(path, scale, bucket)
        self._resolved_raw_paths[entry.id] = path
        self._drivers[path] = driver

        return self._read_crop(arr, driver, offset, shape)

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        path, bucket = self._resolve_seg_paths(entry, organelle)
        arr, driver = self._open_path(path, scale, bucket)
        self._resolved_seg_paths[f"{entry.id}/{organelle}"] = path
        self._drivers[path] = driver

        data = self._read_crop(arr, driver, offset, shape)
        # Keep instance labels as uint32 for proper neuroglancer display
        return np.asarray(data, dtype=np.uint32)

    def get_resolved_raw_path(self, entry_id: str) -> str:
        """Return the actual S3 path used for raw data (zarr or n5)."""
        return self._resolved_raw_paths.get(entry_id, "")

    def get_resolved_seg_path(self, entry_id: str, organelle: str) -> str:
        """Return the actual S3 path used for segmentation data."""
        return self._resolved_seg_paths.get(f"{entry_id}/{organelle}", "")
