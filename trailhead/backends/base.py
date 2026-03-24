"""Abstract base class for data access backends."""

from __future__ import annotations

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
        """Return the (Z, Y, X) shape of the raw volume at the given scale level.

        Args:
            entry: Dataset registry entry.
            scale: Multiscale pyramid level (0 = full resolution).
        """
        ...

    @abstractmethod
    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        """Read a 3D crop from the raw EM volume.

        Args:
            entry: Dataset registry entry.
            offset: (z, y, x) start coordinates.
            shape: (z, y, x) crop size.
            scale: Multiscale pyramid level (0 = full resolution).

        Returns:
            numpy array of shape (z, y, x), dtype uint8.
        """
        ...

    @abstractmethod
    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> NDArray[np.uint8]:
        """Read a 3D crop from a segmentation label volume.

        Args:
            entry: Dataset registry entry.
            organelle: Organelle name (e.g., "mito", "er").
            offset: (z, y, x) start coordinates.
            shape: (z, y, x) crop size.
            scale: Multiscale pyramid level (0 = full resolution).

        Returns:
            numpy array of shape (z, y, x). Non-zero values indicate the organelle.
        """
        ...

    def read_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Read both raw and segmentation crops.

        Returns:
            Tuple of (raw_crop, seg_crop) numpy arrays.
        """
        raw = self.read_raw_crop(entry, offset, shape, scale)
        seg = self.read_segmentation_crop(entry, organelle, offset, shape, scale)
        return raw, seg
