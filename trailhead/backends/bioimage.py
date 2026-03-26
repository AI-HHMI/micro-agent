"""Backend for fluorescence microscopy formats via bioio.

Supports OME-TIFF, CZI (Zeiss), LIF (Leica), and ND2 (Nikon) files using
the bioio library, which normalizes all formats to a consistent (T, C, Z, Y, X)
dimension order with unified metadata access.

Install with: pip install trailhead[fluorescence]
"""

from __future__ import annotations

import threading

import numpy as np
from numpy.typing import NDArray

from trailhead.backends.base import Backend
from trailhead.registry import DatasetEntry

try:
    from bioio import BioImage
except ImportError:
    BioImage = None  # type: ignore[assignment,misc]


def _require_bioio() -> None:
    if BioImage is None:
        raise ImportError(
            "bioio is required for fluorescence format support. "
            "Install with: pip install trailhead[fluorescence]"
        )


class BioImageBackend(Backend):
    """Read fluorescence microscopy data in OME-TIFF, CZI, LIF, ND2 via bioio.

    bioio auto-detects the file format from the extension and normalizes
    all data to (T, C, Z, Y, X) dimension ordering. This backend extracts
    the spatial (Z, Y, X) dimensions for compatibility with the base Backend
    interface, and provides multi-channel access via read_raw_crop_multichannel.
    """

    def __init__(self) -> None:
        self._open_cache: dict[str, "BioImage"] = {}
        self._path_locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # protects _path_locks dict

    def _open(self, path: str) -> "BioImage":
        # Fast path: already cached
        if path in self._open_cache:
            return self._open_cache[path]
        # Get or create a per-path lock so multiple paths can open in parallel
        with self._meta_lock:
            if path not in self._path_locks:
                self._path_locks[path] = threading.Lock()
            lock = self._path_locks[path]
        with lock:
            if path in self._open_cache:
                return self._open_cache[path]
            _require_bioio()
            fs_kwargs = {}
            if path.startswith("s3://") or path.startswith("s3a://"):
                fs_kwargs["anon"] = True
            img = BioImage(path, fs_kwargs=fs_kwargs)
            self._open_cache[path] = img
            return img

    def _resolve_path(self, entry: DatasetEntry) -> str:
        return entry.raw_path or entry.access_url

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        img = self._open(self._resolve_path(entry))
        # bioio dims are (T, C, Z, Y, X); return spatial (Z, Y, X)
        return (img.dims.Z, img.dims.Y, img.dims.X)

    def get_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float]:
        img = self._open(self._resolve_path(entry))
        ps = img.physical_pixel_sizes
        # bioio returns physical sizes in micrometers; convert to nanometers
        z_nm = (ps.Z or 0.0) * 1000.0
        y_nm = (ps.Y or 0.0) * 1000.0
        x_nm = (ps.X or 0.0) * 1000.0

        # Use whatever metadata is available; fill missing dims from known ones
        if y_nm > 0 and x_nm > 0:
            # If Z is missing, assume isotropic with XY (common for light microscopy)
            if z_nm <= 0:
                z_nm = y_nm
            factor = 2 ** scale
            return (z_nm * factor, y_nm * factor, x_nm * factor)

        # Check registry entry as fallback
        if entry.voxel_size_nm and len(entry.voxel_size_nm) >= 3 and any(v > 0 for v in entry.voxel_size_nm[:3]):
            base = entry.voxel_size_nm[:3]
            factor = 2 ** scale
            return (base[0] * factor, base[1] * factor, base[2] * factor)

        # Last resort: default
        return super().get_voxel_size(entry, scale)

    def has_voxel_metadata(self, entry: DatasetEntry) -> bool:
        """Check if bioio can extract physical pixel sizes from the file.

        Returns True if at least Y and X sizes are present (Z may be
        inferred as isotropic if missing).
        """
        try:
            img = self._open(self._resolve_path(entry))
            ps = img.physical_pixel_sizes
            y_nm = (ps.Y or 0.0) * 1000.0
            x_nm = (ps.X or 0.0) * 1000.0
            return y_nm > 0 and x_nm > 0
        except Exception:
            return super().has_voxel_metadata(entry)

    def get_num_scales(self, entry: DatasetEntry) -> int:
        # bioio doesn't expose multiscale pyramids directly; return 1
        return 1

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        """Read channel 0 as a (Z, Y, X) crop at native dtype."""
        img = self._open(self._resolve_path(entry))
        z, y, x = offset
        dz, dy, dx = shape
        data = img.get_image_data(
            "ZYX",
            T=0, C=0,
            Z=slice(z, z + dz),
            Y=slice(y, y + dy),
            X=slice(x, x + dx),
        )
        return np.asarray(data)

    def read_raw_crop_multichannel(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
        channels: list[int] | None = None,
    ) -> NDArray:
        """Read all channels as a (C, Z, Y, X) array at native dtype."""
        img = self._open(self._resolve_path(entry))
        z, y, x = offset
        dz, dy, dx = shape
        data = img.get_image_data(
            "CZYX",
            T=0,
            Z=slice(z, z + dz),
            Y=slice(y, y + dy),
            X=slice(x, x + dx),
        )
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
        """Read segmentation labels. Requires a separate label file in segmentation_paths."""
        seg_path = entry.segmentation_paths.get(organelle)
        if not seg_path:
            raise NotImplementedError(
                f"No segmentation path for organelle '{organelle}' in {entry.id}"
            )
        img = self._open(seg_path)
        z, y, x = offset
        dz, dy, dx = shape
        data = img.get_image_data(
            "ZYX",
            T=0, C=0,
            Z=slice(z, z + dz),
            Y=slice(y, y + dy),
            X=slice(x, x + dx),
        )
        return np.asarray(data)

    def get_channel_metadata(self, entry: DatasetEntry) -> dict:
        """Extract channel names and wavelengths from file metadata."""
        img = self._open(self._resolve_path(entry))
        return {
            "num_channels": img.dims.C,
            "channel_names": list(img.channel_names) if img.channel_names else [],
        }
