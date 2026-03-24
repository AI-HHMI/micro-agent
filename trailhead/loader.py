"""Unified training data loader across microscopy repositories.

Provides a simple iterator that yields (raw_crop, segmentation_crop) pairs
from datasets across multiple repositories, queried by organelle, organism,
or free-text search.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from trailhead.registry import DatasetEntry, Registry
from trailhead.backends.base import Backend
from trailhead.backends.openorganelle import OpenOrganelleBackend


@dataclass
class CropSample:
    """A single training sample."""

    raw: NDArray[np.uint8]
    segmentation: NDArray[np.uint8]
    dataset_id: str
    repository: str
    organelle: str
    offset: tuple[int, int, int]


def _get_backend(repository: str) -> Backend:
    """Return the appropriate backend for a repository."""
    backends: dict[str, type[Backend]] = {
        "OpenOrganelle": OpenOrganelleBackend,
    }
    backend_cls = backends.get(repository)
    if backend_cls is None:
        raise NotImplementedError(
            f"Backend for '{repository}' not yet implemented. "
            f"Available: {list(backends.keys())}"
        )
    return backend_cls()


class UnifiedLoader:
    """Cross-repository training data loader.

    Discovers datasets matching a query, then yields random crops of
    raw + segmentation data suitable for training.

    Usage:
        loader = UnifiedLoader(
            organelle="mito",
            crop_size=(64, 64, 64),
            scale=4,
        )
        for sample in loader:
            print(sample.raw.shape, sample.segmentation.shape)
            # (64, 64, 64) (64, 64, 64)

    Args:
        organelle: Target organelle for segmentation labels (e.g., "mito", "er").
        crop_size: (z, y, x) size of each crop.
        scale: Multiscale pyramid level (0 = full res, higher = downsampled).
        query: Additional free-text search to filter datasets.
        organism: Filter by organism.
        cell_type: Filter by cell type.
        repositories: Restrict to specific repositories (default: all available).
        num_samples: Total samples to yield (default: unlimited / 1000).
        seed: Random seed for reproducibility.
        registry: Custom registry (default: built-in catalog).
    """

    def __init__(
        self,
        organelle: str,
        crop_size: tuple[int, int, int] = (64, 64, 64),
        scale: int = 0,
        query: str = "",
        organism: str = "",
        cell_type: str = "",
        repositories: list[str] | None = None,
        num_samples: int = 1000,
        seed: int | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.organelle = organelle
        self.crop_size = crop_size
        self.scale = scale
        self.num_samples = num_samples

        self._rng = random.Random(seed)
        self._registry = registry or Registry()

        # Find matching datasets
        search_kwargs: dict = {
            "organelle": organelle,
            "has_segmentation": True,
        }
        if query:
            search_kwargs["query"] = query
        if organism:
            search_kwargs["organism"] = organism
        if cell_type:
            search_kwargs["cell_type"] = cell_type
        if repositories:
            # Search per-repo and combine
            self._datasets: list[DatasetEntry] = []
            for repo in repositories:
                self._datasets.extend(
                    self._registry.search(**search_kwargs, repository=repo)
                )
        else:
            self._datasets = self._registry.search(**search_kwargs)

        if not self._datasets:
            raise ValueError(
                f"No datasets found matching organelle='{organelle}', "
                f"query='{query}', organism='{organism}'. "
                f"Available organelles: {self._registry.list_organelles()}"
            )

        # Initialize backends (one per repository type)
        self._backends: dict[str, Backend] = {}
        for entry in self._datasets:
            if entry.repository not in self._backends:
                self._backends[entry.repository] = _get_backend(entry.repository)

        # Cache volume shapes for random offset generation
        self._shapes: dict[str, tuple[int, ...]] = {}

    def _get_shape(self, entry: DatasetEntry) -> tuple[int, ...]:
        if entry.id not in self._shapes:
            backend = self._backends[entry.repository]
            self._shapes[entry.id] = backend.get_volume_shape(entry, self.scale)
        return self._shapes[entry.id]

    def _random_offset(self, entry: DatasetEntry) -> tuple[int, int, int]:
        vol_shape = self._get_shape(entry)
        cz, cy, cx = self.crop_size
        max_z = max(0, vol_shape[0] - cz)
        max_y = max(0, vol_shape[1] - cy)
        max_x = max(0, vol_shape[2] - cx)
        return (
            self._rng.randint(0, max_z),
            self._rng.randint(0, max_y),
            self._rng.randint(0, max_x),
        )

    def __iter__(self) -> Iterator[CropSample]:
        for _ in range(self.num_samples):
            entry = self._rng.choice(self._datasets)
            backend = self._backends[entry.repository]
            offset = self._random_offset(entry)

            raw, seg = backend.read_crop(
                entry, self.organelle, offset, self.crop_size, self.scale
            )

            yield CropSample(
                raw=raw,
                segmentation=seg,
                dataset_id=entry.id,
                repository=entry.repository,
                organelle=self.organelle,
                offset=offset,
            )

    def __len__(self) -> int:
        return self.num_samples

    @property
    def datasets(self) -> list[DatasetEntry]:
        """Return the list of datasets this loader will sample from."""
        return list(self._datasets)

    def summary(self) -> str:
        """Return a human-readable summary of what this loader will do."""
        lines = [
            f"UnifiedLoader: {self.num_samples} crops of {self.crop_size} at scale s{self.scale}",
            f"  Organelle: {self.organelle}",
            f"  Datasets ({len(self._datasets)}):",
        ]
        for entry in self._datasets:
            lines.append(f"    - {entry.id} ({entry.repository}) — {entry.title}")
        return "\n".join(lines)
