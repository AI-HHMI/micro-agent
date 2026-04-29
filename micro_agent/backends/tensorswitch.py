"""TensorSwitch backend for reading local OME-Zarr files.

Provides Backend ABC implementation that reads TensorSwitch-converted
datasets (OME-Zarr v3 with nested raw/labels structure, pyramids).
Supports any format TensorSwitch can read via its tiered reader system.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from micro_agent.backends.base import Backend
from micro_agent.registry import DatasetEntry

try:
    from tensorswitch_v2.api.readers import Readers
    from tensorswitch_v2.utils.metadata_utils import (
        auto_detect_max_level,
        detect_level_format,
        get_level_name,
    )

    _HAS_TENSORSWITCH = True
except ImportError:
    _HAS_TENSORSWITCH = False


class TensorSwitchBackend(Backend):
    """Backend for reading local datasets via TensorSwitch.

    Reads OME-Zarr v3 containers with nested structure (raw/s0, labels/organelle/s0)
    as well as flat zarr, N5, and any other format supported by TensorSwitch.

    The backend expects ``entry.raw_path`` to point to the zarr container root
    (e.g. ``/data/jrc_hela-2.zarr``). Pyramid levels are auto-detected.
    """

    def __init__(self) -> None:
        if not _HAS_TENSORSWITCH:
            raise ImportError(
                "TensorSwitchBackend requires tensorswitch_v2. "
                "Install with: pip install tensorswitch"
            )
        # Cache: (container_path, group_subpath, scale) -> TensorStore
        self._store_cache: dict[tuple[str, str, int], object] = {}

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_group_path(self, container: str) -> str:
        """Find the image group path within a container (e.g. 'raw' or '')."""
        raw_path = os.path.join(container, "raw")
        if os.path.isdir(raw_path):
            return "raw"
        return ""

    def _resolve_seg_group_path(self, container: str, organelle: str) -> str:
        """Find the segmentation group path within a container."""
        # Check labels/{organelle}
        labels_path = os.path.join(container, "labels", organelle)
        if os.path.isdir(labels_path):
            return f"labels/{organelle}"
        # Check labels/segmentation
        seg_path = os.path.join(container, "labels", "segmentation")
        if os.path.isdir(seg_path):
            return "labels/segmentation"
        return ""

    def _open_store(self, container: str, group: str, scale: int):
        """Open a TensorStore for a specific scale level, with caching."""
        cache_key = (container, group, scale)
        if cache_key in self._store_cache:
            return self._store_cache[cache_key]

        # Determine level name
        group_path = os.path.join(container, group) if group else container
        prefix = detect_level_format(group_path)
        level_name = get_level_name(scale, prefix)

        # Open the reader using the group path as container so the OME
        # multiscale metadata in group/zarr.json is found by the fallback.
        level_full_path = os.path.join(group_path, level_name)
        if os.path.isfile(os.path.join(level_full_path, "zarr.json")):
            reader = Readers.zarr3(group_path, dataset_path=level_name)
        elif os.path.isfile(os.path.join(level_full_path, ".zarray")):
            reader = Readers.zarr2(group_path, dataset_path=level_name)
        else:
            reader = Readers.auto_detect(level_full_path)

        store = reader.get_tensorstore()
        self._store_cache[cache_key] = store

        # Also cache the reader for voxel size queries
        reader_key = ("_reader", group, scale)
        self._store_cache[reader_key] = reader

        return store

    def _get_reader(self, container: str, group: str, scale: int):
        """Get the cached reader for voxel size queries."""
        reader_key = ("_reader", group, scale)
        if reader_key not in self._store_cache:
            # Opening the store also caches the reader
            self._open_store(container, group, scale)
        return self._store_cache[reader_key]

    def _get_max_level(self, container: str, group: str) -> int:
        """Get the maximum pyramid level for a group."""
        group_path = os.path.join(container, group) if group else container
        max_level, _ = auto_detect_max_level(group_path)
        return max_level if max_level is not None else 0

    @staticmethod
    def _read_ome_voxel_sizes(
        group_path: str, level_name: str,
    ) -> tuple[float, float, float] | None:
        """Read voxel sizes from OME-NGFF multiscale metadata in group zarr.json.

        The TensorSwitch reader's fallback only triggers when the array-level
        zarr.json is missing, but for OME-Zarr v3 the array file always exists.
        This method reads the group-level metadata directly.
        """
        from tensorswitch_v2.utils.format_loaders import convert_to_nanometers

        zarr_json = os.path.join(group_path, "zarr.json")
        if not os.path.isfile(zarr_json):
            return None
        try:
            with open(zarr_json) as f:
                meta = json.load(f)
        except Exception:
            return None

        attrs = meta.get("attributes", {})
        # OME-NGFF v0.5: multiscales under ome.multiscales
        multiscales = attrs.get("ome", {}).get("multiscales", [])
        if not multiscales:
            # Older: multiscales at top level
            multiscales = attrs.get("multiscales", [])
        if not multiscales:
            return None

        ms = multiscales[0]
        axes = ms.get("axes", [])
        for ds in ms.get("datasets", []):
            if ds.get("path") == level_name:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        scales = t["scale"]
                        voxel = {"z": 1.0, "y": 1.0, "x": 1.0}
                        for i, ax in enumerate(axes):
                            name = ax.get("name", "").lower()
                            if i < len(scales) and name in voxel:
                                unit = ax.get("unit", "micrometer")
                                voxel[name] = convert_to_nanometers(
                                    scales[i], unit,
                                )
                        return (voxel["z"], voxel["y"], voxel["x"])
        return None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_volume_shape(
        self, entry: DatasetEntry, scale: int = 0,
    ) -> tuple[int, ...]:
        """Return (Z, Y, X) shape at the given scale level."""
        container = entry.raw_path or entry.access_url
        group = self._resolve_group_path(container)
        store = self._open_store(container, group, scale)
        return tuple(store.shape)

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        """Read a 3D crop from the raw volume."""
        container = entry.raw_path or entry.access_url
        group = self._resolve_group_path(container)
        store = self._open_store(container, group, scale)

        oz, oy, ox = offset
        sz, sy, sx = shape
        data = store[oz:oz + sz, oy:oy + sy, ox:ox + sx].read().result()
        return np.asarray(data)

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        """Read a 3D crop from a segmentation label volume."""
        container = entry.raw_path or entry.access_url
        group = self._resolve_seg_group_path(container, organelle)
        if not group:
            raise FileNotFoundError(
                f"No segmentation found for organelle '{organelle}' "
                f"in {container}"
            )
        store = self._open_store(container, group, scale)

        oz, oy, ox = offset
        sz, sy, sx = shape
        data = store[oz:oz + sz, oy:oy + sy, ox:ox + sx].read().result()
        return np.asarray(data).astype(np.uint32)

    # ------------------------------------------------------------------
    # Override methods for better metadata
    # ------------------------------------------------------------------

    def _voxel_size_for_group(
        self, container: str, group: str, scale: int,
    ) -> tuple[float, float, float]:
        """Read voxel sizes from OME group metadata, falling back to reader."""
        group_path = os.path.join(container, group) if group else container
        prefix = detect_level_format(group_path)
        level_name = get_level_name(scale, prefix)

        # Try OME metadata from group zarr.json first
        ome = self._read_ome_voxel_sizes(group_path, level_name)
        if ome is not None:
            return ome

        # Fall back to reader's get_voxel_sizes()
        reader = self._get_reader(container, group, scale)
        voxel = reader.get_voxel_sizes()
        return (voxel.get("z", 1.0), voxel.get("y", 1.0), voxel.get("x", 1.0))

    def get_voxel_size(
        self, entry: DatasetEntry, scale: int = 0,
    ) -> tuple[float, float, float]:
        """Return (z, y, x) voxel size in nm from OME-NGFF metadata."""
        container = entry.raw_path or entry.access_url
        group = self._resolve_group_path(container)
        return self._voxel_size_for_group(container, group, scale)

    def has_voxel_metadata(self, entry: DatasetEntry) -> bool:
        """TensorSwitch OME-Zarr always has voxel size metadata."""
        return True

    def pick_scale(
        self,
        entry: DatasetEntry,
        target_nm: tuple[float, float, float],
    ) -> int:
        """Pick the coarsest scale finer than target, using actual per-level metadata."""
        container = entry.raw_path or entry.access_url
        group = self._resolve_group_path(container)
        max_level = self._get_max_level(container, group)

        best = 0
        for s in range(max_level + 1):
            try:
                vox = self.get_voxel_size(entry, s)
            except Exception:
                break
            if all(v <= t for v, t in zip(vox, target_nm)):
                best = s
            else:
                break
        return best

    def get_seg_voxel_size(
        self, entry: DatasetEntry, organelle: str, scale: int = 0,
    ) -> tuple[float, float, float]:
        """Return segmentation voxel size from its own metadata."""
        container = entry.raw_path or entry.access_url
        group = self._resolve_seg_group_path(container, organelle)
        if not group:
            return self.get_voxel_size(entry, scale)
        return self._voxel_size_for_group(container, group, scale)
