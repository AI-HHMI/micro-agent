"""Abstract base class for data access backends."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from trailhead.registry import DatasetEntry


class Backend(ABC):
    """Base class for repository-specific data access.

    Each backend knows how to read image and segmentation data from a
    specific repository's storage format (S3/N5, OME-Zarr, OMERO, etc).
    """

    @abstractmethod
    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        """Return the (Z, Y, X) shape of the raw volume at the given scale level."""
        ...

    @abstractmethod
    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        """Read a 3D crop from the raw volume at native dtype."""
        ...

    @abstractmethod
    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray:
        """Read a 3D crop from a segmentation label volume."""
        ...

    # Sentinel default used when no voxel size metadata is found.
    # Deliberately not 8nm (common EM value) so it's clear when it's a guess.
    _DEFAULT_VOXEL_NM = (10.0, 10.0, 10.0)

    def get_voxel_size(self, entry: DatasetEntry, scale: int = 0) -> tuple[float, float, float]:
        """Return (z, y, x) voxel size in nm at the given scale level.

        Default implementation uses entry.voxel_size_nm with 2x per scale level.
        Subclasses should override to read from volume metadata.
        """
        if entry.voxel_size_nm and len(entry.voxel_size_nm) >= 3 and any(v > 0 for v in entry.voxel_size_nm[:3]):
            base = entry.voxel_size_nm[:3]
        else:
            base = list(self._DEFAULT_VOXEL_NM)
        factor = 2 ** scale
        return (base[0] * factor, base[1] * factor, base[2] * factor)

    def has_voxel_metadata(self, entry: DatasetEntry) -> bool:
        """Check if real voxel size metadata exists for this entry.

        Subclasses that read voxel sizes from file metadata should override.
        Returns True if the voxel size returned by get_voxel_size came from
        real metadata, not defaults.
        """
        return bool(
            entry.voxel_size_nm
            and len(entry.voxel_size_nm) >= 3
            and any(v > 0 for v in entry.voxel_size_nm[:3])
        )

    def get_num_scales(self, entry: DatasetEntry) -> int:
        """Return the number of available scale levels.

        Default: try opening scales until one fails, max 10.
        Subclasses should override if they can determine this from metadata.
        """
        return 6  # safe default for most multiscale pyramids

    def pick_scale(
        self,
        entry: DatasetEntry,
        target_nm: tuple[float, float, float],
    ) -> int:
        """Pick the coarsest scale level that's still finer than target_nm.

        Returns the scale index. If all scales are coarser than target,
        returns 0 (finest available).
        """
        num = self.get_num_scales(entry)
        best = 0
        for s in range(num):
            vox = self.get_voxel_size(entry, s)
            if all(v <= t for v, t in zip(vox, target_nm)):
                best = s
            else:
                break
        return best

    def read_raw_crop_multichannel(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
        channels: list[int] | None = None,
    ) -> NDArray:
        """Read a multi-channel 3D crop. Returns array of shape (C, Z, Y, X).

        Default implementation wraps read_raw_crop for single-channel backends.
        Fluorescence backends should override to return all channels at native dtype.
        """
        data = self.read_raw_crop(entry, offset, shape, scale)
        result = data[np.newaxis, ...]  # (1, Z, Y, X)
        if channels is not None:
            result = result[channels]
        return result

    def read_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> tuple[NDArray, NDArray]:
        """Read both raw and segmentation crops."""
        raw = self.read_raw_crop(entry, offset, shape, scale)
        seg = self.read_segmentation_crop(entry, organelle, offset, shape, scale)
        return raw, seg
