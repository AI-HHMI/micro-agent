"""Unified training data loader across microscopy repositories.

Provides a simple iterator that yields (raw_crop, segmentation_crop) pairs
from datasets across multiple repositories, queried by organelle, organism,
or free-text search.
"""

from __future__ import annotations

import math
import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from trailhead.registry import DatasetEntry, Registry
from trailhead.backends.base import Backend
from trailhead.backends.openorganelle import OpenOrganelleBackend
from trailhead.backends.microns import MICrONSBackend
from trailhead.backends.empiar import EMPIARBackend
from trailhead.backends.idr import IDRBackend


@dataclass
class CropSample:
    """A single training sample."""

    raw: NDArray[np.uint8]
    segmentation: NDArray[np.uint8] | None
    dataset_id: str
    repository: str
    organelle: str
    offset: tuple[int, int, int]
    resolution_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_resolution_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_used: int = 0
    seg_status: str = "no_seg_available"  # "loaded", "empty", "failed: ...", "no_seg_available"
    raw_path: str = ""
    seg_path: str = ""


def _get_backend(repository: str) -> Backend:
    """Return the appropriate backend for a repository."""
    backends: dict[str, type[Backend]] = {
        "OpenOrganelle": OpenOrganelleBackend,
        "MICrONS": MICrONSBackend,
        "FlyEM": MICrONSBackend,
        "Google": MICrONSBackend,
        "OpenNeuroData": MICrONSBackend,
        "EMPIAR": EMPIARBackend,
        "IDR": IDRBackend,
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
            resolution_nm=(8.0, 8.0, 8.0),
        )
        for sample in loader:
            print(sample.raw.shape, sample.segmentation.shape)
            # (64, 64, 64) (64, 64, 64)

    Args:
        organelle: Target organelle for segmentation labels (e.g., "mito", "er").
        crop_size: (z, y, x) size of each crop in output voxels.
        resolution_nm: Target (z, y, x) voxel size in nanometers. The loader
            picks the best multiscale level and resamples to this resolution.
            If None, reads at native resolution (scale 0).
        query: Additional free-text search to filter datasets.
        organism: Filter by organism.
        cell_type: Filter by cell type.
        repositories: Restrict to specific repositories (default: all available).
        num_samples: Total samples to yield (default: 1000).
        seed: Random seed for reproducibility.
        registry: Custom registry (default: built-in catalog).
        require_segmentation: If True, only use datasets with segmentations.
        balance_repositories: If True, sample equally across repositories.
    """

    def __init__(
        self,
        organelle: str = "",
        crop_size: tuple[int, int, int] = (64, 64, 64),
        resolution_nm: tuple[float, float, float] | None = None,
        query: str = "",
        organism: str = "",
        cell_type: str = "",
        repositories: list[str] | None = None,
        num_samples: int = 1000,
        seed: int | None = None,
        registry: Registry | None = None,
        require_segmentation: bool = False,
        balance_repositories: bool = False,
    ) -> None:
        self.organelle = organelle
        self.crop_size = crop_size
        self.resolution_nm = resolution_nm
        self.num_samples = num_samples
        self._require_segmentation = require_segmentation
        self._balance_repositories = balance_repositories

        self._rng = random.Random(seed)
        self._registry = registry or Registry()

        # Find matching datasets
        search_kwargs: dict = {}
        if organelle and require_segmentation:
            search_kwargs["organelle"] = organelle
        if require_segmentation:
            search_kwargs["has_segmentation"] = True
        if query:
            search_kwargs["query"] = query
        if organism:
            search_kwargs["organism"] = organism
        if cell_type:
            search_kwargs["cell_type"] = cell_type
        if repositories:
            self._datasets: list[DatasetEntry] = []
            for repo in repositories:
                self._datasets.extend(
                    self._registry.search(**search_kwargs, repository=repo)
                )
        else:
            self._datasets = self._registry.search(**search_kwargs)

        # When not requiring segmentation but organelle is specified,
        # filter to datasets that have the organelle or have no metadata.
        if organelle and not require_segmentation:
            organelle_lower = organelle.lower()
            self._datasets = [
                e for e in self._datasets
                if any(organelle_lower in o.lower() for o in e.organelles)
                or not e.organelles
            ]

        # Exclude datasets that don't support random access
        skipped = [e for e in self._datasets if not e.supports_random_access]
        self._datasets = [e for e in self._datasets if e.supports_random_access]
        if skipped:
            print(f"  Skipped {len(skipped)} dataset(s) requiring download: "
                  f"{', '.join(e.id for e in skipped)}")

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

        # Group datasets by repository for balanced sampling
        self._by_repo: dict[str, list[DatasetEntry]] = {}
        for entry in self._datasets:
            self._by_repo.setdefault(entry.repository, []).append(entry)

        # Cache per-dataset: (scale, source_voxel_size, zoom_factors, read_shape)
        self._scale_info: dict[str, tuple[int, tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]] = {}
        # Cache volume shapes at the chosen scale
        self._shapes: dict[str, tuple[int, ...]] = {}

    def _get_scale_info(self, entry: DatasetEntry) -> tuple[int, tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]:
        """Return (scale, source_voxel_nm, zoom_factors, read_shape) for a dataset."""
        if entry.id in self._scale_info:
            return self._scale_info[entry.id]

        backend = self._backends[entry.repository]

        if self.resolution_nm is None:
            # No target resolution — read at scale 0, no resampling
            vox = backend.get_voxel_size(entry, 0)
            info = (0, vox, (1.0, 1.0, 1.0), self.crop_size)
            self._scale_info[entry.id] = info
            return info

        target = self.resolution_nm
        scale = backend.pick_scale(entry, target)
        src_vox = backend.get_voxel_size(entry, scale)

        # zoom_factors: applied to read data to produce output at target resolution
        # zoom < 1 means downsampling (source finer than target)
        # zoom > 1 means upsampling (source coarser than target)
        zoom_factors = (src_vox[0] / target[0], src_vox[1] / target[1], src_vox[2] / target[2])

        # How many source voxels to read to cover crop_size output voxels
        read_shape = (
            math.ceil(self.crop_size[0] / zoom_factors[0]),
            math.ceil(self.crop_size[1] / zoom_factors[1]),
            math.ceil(self.crop_size[2] / zoom_factors[2]),
        )

        info = (scale, src_vox, zoom_factors, read_shape)
        self._scale_info[entry.id] = info
        return info

    def _get_shape(self, entry: DatasetEntry) -> tuple[int, ...]:
        if entry.id not in self._shapes:
            backend = self._backends[entry.repository]
            scale, _, _, _ = self._get_scale_info(entry)
            self._shapes[entry.id] = backend.get_volume_shape(entry, scale)
        return self._shapes[entry.id]

    def _clamp_read_shape(self, entry: DatasetEntry) -> tuple[int, int, int]:
        """Return the read shape clamped to the volume dimensions."""
        vol_shape = self._get_shape(entry)
        _, _, _, read_shape = self._get_scale_info(entry)
        return (
            min(read_shape[0], vol_shape[0]),
            min(read_shape[1], vol_shape[1]),
            min(read_shape[2], vol_shape[2]),
        )

    def _random_offset(self, entry: DatasetEntry) -> tuple[int, int, int]:
        vol_shape = self._get_shape(entry)
        read = self._clamp_read_shape(entry)
        max_z = max(0, vol_shape[0] - read[0])
        max_y = max(0, vol_shape[1] - read[1])
        max_x = max(0, vol_shape[2] - read[2])
        return (
            self._rng.randint(0, max_z),
            self._rng.randint(0, max_y),
            self._rng.randint(0, max_x),
        )

    def _resample(self, data: np.ndarray, zoom_factors: tuple[float, float, float], order: int = 1) -> np.ndarray:
        """Resample data to target resolution. order=0 for segmentation, 1 for raw."""
        if all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
            return data  # no resampling needed

        resampled = zoom(data.astype(np.float32), zoom_factors, order=order)
        # Trim or pad to exact crop_size
        out = np.zeros(self.crop_size, dtype=data.dtype)
        slices = tuple(slice(0, min(s, c)) for s, c in zip(resampled.shape, self.crop_size))
        out[slices] = resampled[slices].astype(data.dtype)
        return out

    def _fetch_one(self) -> CropSample | None:
        """Fetch a single random crop. Returns None on failure."""
        if self._balance_repositories:
            repo = self._rng.choice(list(self._by_repo.keys()))
            entry = self._rng.choice(self._by_repo[repo])
        else:
            entry = self._rng.choice(self._datasets)
        backend = self._backends[entry.repository]

        try:
            scale, src_vox, zoom_factors, _ = self._get_scale_info(entry)
            offset = self._random_offset(entry)
            read = self._clamp_read_shape(entry)
            raw = backend.read_raw_crop(entry, offset, read, scale)
        except Exception as e:
            print(f"  Skipping {entry.id} ({entry.repository}): {e}")
            return None

        # Resample to target resolution
        raw = self._resample(raw, zoom_factors, order=1)

        seg = None
        seg_status = "no_seg_available"
        if entry.has_segmentation and self.organelle in entry.segmentation_paths:
            try:
                seg = backend.read_segmentation_crop(
                    entry, self.organelle, offset, read, scale
                )
                seg = self._resample(seg, zoom_factors, order=0)
                seg_status = "loaded" if seg.any() else "empty"
            except Exception as e:
                seg_status = f"failed: {e}"

        # Resolve paths for display
        raw_path = ""
        seg_path = ""
        if hasattr(backend, 'get_resolved_raw_path'):
            resolved = backend.get_resolved_raw_path(entry.id)
            if resolved:
                raw_path = f"s3://janelia-cosem-datasets/{resolved}"
        if not raw_path:
            raw_path = (entry.access_url + entry.raw_path) if entry.raw_path else entry.access_url
        if self.organelle and hasattr(backend, 'get_resolved_seg_path'):
            resolved = backend.get_resolved_seg_path(entry.id, self.organelle)
            if resolved:
                seg_path = f"s3://janelia-cosem-datasets/{resolved}"
        if not seg_path and self.organelle and self.organelle in entry.segmentation_paths:
            seg_path = (entry.access_url + entry.segmentation_paths[self.organelle])

        return CropSample(
            raw=raw,
            segmentation=seg,
            dataset_id=entry.id,
            repository=entry.repository,
            organelle=self.organelle,
            offset=offset,
            resolution_nm=self.resolution_nm or src_vox,
            source_resolution_nm=src_vox,
            scale_used=scale,
            seg_status=seg_status,
            raw_path=raw_path,
            seg_path=seg_path,
        )

    def __iter__(self) -> Iterator[CropSample]:
        for _ in range(self.num_samples):
            sample = self._fetch_one()
            if sample is not None:
                yield sample

    def prefetch_iter(self, prefetch: int = 3) -> Iterator[CropSample]:
        """Iterator that prefetches crops in background threads."""
        buf: queue.Queue[CropSample | None] = queue.Queue(maxsize=prefetch)
        stop = threading.Event()

        def _worker() -> None:
            produced = 0
            while produced < self.num_samples and not stop.is_set():
                sample = self._fetch_one()
                if sample is not None:
                    buf.put(sample)
                    produced += 1
            buf.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                sample = buf.get()
                if sample is None:
                    break
                yield sample
        finally:
            stop.set()

    def __len__(self) -> int:
        return self.num_samples

    @property
    def datasets(self) -> list[DatasetEntry]:
        """Return the list of datasets this loader will sample from."""
        return list(self._datasets)

    def summary(self) -> str:
        """Return a human-readable summary of what this loader will do."""
        res_str = f"{self.resolution_nm} nm" if self.resolution_nm else "native"
        lines = [
            f"UnifiedLoader: {self.num_samples} crops of {self.crop_size} at {res_str}",
            f"  Organelle: {self.organelle or '(any)'}",
            f"  Datasets ({len(self._datasets)}):",
        ]
        for entry in self._datasets:
            seg_flag = " [+seg]" if entry.has_segmentation else " [raw only]"
            try:
                scale, src_vox, zoom_factors, read_shape = self._get_scale_info(entry)
                vox_str = f"  native@s{scale}: {src_vox[0]:.1f}x{src_vox[1]:.1f}x{src_vox[2]:.1f}nm"
                if any(abs(z - 1.0) > 1e-6 for z in zoom_factors):
                    vox_str += f" → resample {zoom_factors[0]:.2f}x"
            except Exception:
                vox_str = ""
            lines.append(f"    - {entry.id} ({entry.repository}){seg_flag}{vox_str}")
        return "\n".join(lines)
