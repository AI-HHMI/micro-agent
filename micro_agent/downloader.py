"""Download contiguous subvolumes from microscopy repositories to local zarr.

Usage:
    python -m micro_agent.downloader /path/to/save \
        --organelle mito \
        --resolution-nm 8 8 8 \
        --max-size-gb 5.0 \
        --require-segmentation
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorswitch_v2.writers import Zarr3Writer

from micro_agent.backends.base import Backend
from micro_agent.loader import _get_backend
from micro_agent.registry import DatasetEntry, Registry


@dataclass
class DownloadedVolume:
    """Metadata for one downloaded subvolume."""

    zarr_path: str
    dataset_id: str
    repository: str
    organelle: str
    offset: list[int]
    shape: list[int]
    resolution_nm: list[float]
    scale_used: int
    has_segmentation: bool
    has_pyramid: bool
    raw_dtype: str
    size_bytes: int
    raw_source: str


@dataclass
class DownloadReport:
    num_volumes: int
    num_datasets: int
    total_bytes: int
    manifest_path: str


class DataDownloader:
    """Download contiguous subvolumes from matching datasets to local zarr.

    Discovers datasets matching the given filters, then downloads a centered
    subvolume from each, distributing a total size budget evenly across
    datasets.

    Output is a single OME-Zarr v3 container per dataset with nested structure:
    - ``{dataset_id}.zarr/raw/s0`` for raw image data
    - ``{dataset_id}.zarr/labels/{organelle}/s0`` for segmentation

    With ``generate_pyramid=True``, multiscale pyramid levels (s1, s2, ...)
    are generated automatically after download.

    Args:
        save_path: Root directory for downloaded data.
        organelle: Filter by organelle (e.g. "mito", "er").
        resolution_nm: Target (z, y, x) voxel size in nm. The downloader
            picks the coarsest scale still finer than this. If None, uses
            native resolution (scale 0).
        max_size_gb: Approximate total download budget in gigabytes.
        repositories: Restrict to specific repositories (default: all).
        require_segmentation: If True, only use datasets with segmentations.
        seed: Random seed (unused currently, reserved for future offset jitter).
        modality_class: Filter by modality ("em", "fluorescence", or "").
        slab_size: Number of Z-slices to read/write at a time. Controls
            peak memory usage.
        generate_pyramid: If True, generate multiscale pyramid after download.
    """

    _EXCLUDED_REPOS = {"EMPIAR"}

    def __init__(
        self,
        save_path: str | Path,
        organelle: str = "",
        resolution_nm: tuple[float, float, float] | None = None,
        max_size_gb: float = 1.0,
        repositories: list[str] | None = None,
        require_segmentation: bool = False,
        seed: int | None = None,
        modality_class: str = "",
        slab_size: int = 64,
        generate_pyramid: bool = False,
    ) -> None:
        self.save_path = Path(save_path)
        self.organelle = organelle
        self.resolution_nm = resolution_nm
        self.max_size_gb = max_size_gb
        self.require_segmentation = require_segmentation
        self.seed = seed
        self.modality_class = modality_class
        self.slab_size = slab_size
        self.generate_pyramid = generate_pyramid

        self._registry = Registry()
        self._datasets = self._find_datasets(organelle, repositories, require_segmentation, modality_class)
        self._backends: dict[str, Backend] = {}
        for entry in self._datasets:
            if entry.repository not in self._backends:
                self._backends[entry.repository] = _get_backend(entry.repository)

    def _find_datasets(
        self,
        organelle: str,
        repositories: list[str] | None,
        require_segmentation: bool,
        modality_class: str,
    ) -> list[DatasetEntry]:
        """Find datasets matching filters (mirrors UnifiedLoader logic)."""
        search_kwargs: dict = {}
        if organelle and require_segmentation:
            search_kwargs["organelle"] = organelle
        if require_segmentation:
            search_kwargs["has_segmentation"] = True
        if modality_class:
            search_kwargs["modality_class"] = modality_class

        if repositories:
            datasets: list[DatasetEntry] = []
            for repo in repositories:
                datasets.extend(self._registry.search(**search_kwargs, repository=repo))
        else:
            datasets = self._registry.search(**search_kwargs)

        # Filter by organelle when not requiring segmentation
        if organelle and not require_segmentation:
            organelle_lower = organelle.lower()
            datasets = [
                e for e in datasets
                if any(organelle_lower in o.lower() for o in e.organelles)
                or not e.organelles
            ]

        # Exclude slow/unreliable repos and datasets without random access
        datasets = [e for e in datasets if e.repository not in self._EXCLUDED_REPOS]
        skipped = [e for e in datasets if not e.supports_random_access]
        datasets = [e for e in datasets if e.supports_random_access]
        if skipped:
            print(
                f"  Skipped {len(skipped)} dataset(s) without direct data paths: "
                f"{', '.join(e.id for e in skipped[:5])}"
                f"{'...' if len(skipped) > 5 else ''}"
            )

        if not datasets:
            raise ValueError(
                f"No datasets found matching organelle='{organelle}', "
                f"repositories={repositories}. "
                f"Available organelles: {self._registry.list_organelles()}"
            )

        print(f"Found {len(datasets)} matching dataset(s)")
        return datasets

    def _pick_scale_and_voxel(
        self, entry: DatasetEntry, backend: Backend,
    ) -> tuple[int, tuple[float, float, float]]:
        """Pick scale level and return (scale, voxel_size_nm)."""
        if self.resolution_nm is None:
            vox = backend.get_voxel_size(entry, 0)
            return 0, vox
        scale = backend.pick_scale(entry, self.resolution_nm)
        vox = backend.get_voxel_size(entry, scale)
        return scale, vox

    def _compute_subvolume(
        self,
        vol_shape: tuple[int, ...],
        budget_bytes: int,
        bytes_per_voxel: int,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Compute a centered subvolume that fits within the byte budget.

        Returns (offset, shape) both as (z, y, x).
        """
        vz, vy, vx = vol_shape[0], vol_shape[1], vol_shape[2]
        total_vol_bytes = vz * vy * vx * bytes_per_voxel

        # If the full volume fits, download it all
        if total_vol_bytes <= budget_bytes:
            return (0, 0, 0), (vz, vy, vx)

        # Compute isotropic cube side that fits in budget
        max_voxels = budget_bytes / bytes_per_voxel
        side = int(max_voxels ** (1.0 / 3.0))

        # Clamp to volume dimensions
        sz = min(side, vz)
        sy = min(side, vy)
        sx = min(side, vx)

        # If clamping freed up budget, redistribute to unclamped dims
        # Iterate a couple times to converge
        for _ in range(3):
            remaining = max_voxels / max(sz * sy * sx, 1) * (sz * sy * sx)
            # Actually recompute: given clamped dims, maximize unclamped
            clamped_product = 1
            unclamped_dims = []
            for s, v in [(sz, vz), (sy, vy), (sx, vx)]:
                if s >= v:
                    clamped_product *= v
                else:
                    unclamped_dims.append(v)

            if not unclamped_dims:
                break

            remaining_voxels = max_voxels / clamped_product
            new_side = int(remaining_voxels ** (1.0 / len(unclamped_dims)))
            sz = min(new_side, vz) if sz < vz else sz
            sy = min(new_side, vy) if sy < vy else sy
            sx = min(new_side, vx) if sx < vx else sx

        # Center the subvolume
        oz = max(0, (vz - sz) // 2)
        oy = max(0, (vy - sy) // 2)
        ox = max(0, (vx - sx) // 2)

        return (oz, oy, ox), (sz, sy, sx)

    def _download_volume(
        self,
        entry: DatasetEntry,
        backend: Backend,
        scale: int,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        voxel_nm: tuple[float, float, float],
        container_path: Path,
        label: str = "raw",
        organelle: str = "",
    ) -> int:
        """Download a subvolume in Z-slabs via tensorswitch Zarr3Writer.

        Writes into a single OME-Zarr container with nested structure:
        - raw data: container_path/raw/s0
        - segmentation: container_path/labels/{organelle}/s0

        Returns bytes written to disk.
        """
        is_seg = label != "raw"
        oz, oy, ox = offset
        sz, sy, sx = shape

        # Probe dtype from a tiny read
        probe = (
            backend.read_segmentation_crop(entry, organelle, offset, (1, 1, 1), scale)
            if is_seg
            else backend.read_raw_crop(entry, offset, (1, 1, 1), scale)
        )
        dtype = str(probe.dtype)

        writer = Zarr3Writer(
            str(container_path),
            use_sharding=True,
            compression="zstd",
            compression_level=3,
            data_type="labels" if is_seg else "image",
            image_key="raw",
            label_key=organelle if organelle else "segmentation",
        )
        # chunk_shape=None lets TensorSwitch auto-calculate (default 64³ for spatial)
        spec = writer.create_output_spec(
            shape=(sz, sy, sx),
            dtype=dtype,
            chunk_shape=None,
        )
        store = writer.open_store(spec, create=True, delete_existing=True)

        for z_start in range(0, sz, self.slab_size):
            z_end = min(z_start + self.slab_size, sz)
            slab_shape = (z_end - z_start, sy, sx)
            slab_offset = (oz + z_start, oy, ox)

            if is_seg:
                slab = backend.read_segmentation_crop(
                    entry, organelle, slab_offset, slab_shape, scale,
                )
            else:
                slab = backend.read_raw_crop(entry, slab_offset, slab_shape, scale)

            store[z_start:z_end] = slab
            print(f"    {label} z=[{z_start}:{z_end}]/{sz}")

        # Write OME-NGFF metadata with voxel sizes
        writer.write_metadata(
            voxel_sizes={"z": voxel_nm[0], "y": voxel_nm[1], "x": voxel_nm[2]},
            image_name=f"{entry.id}_{label}",
            is_label=is_seg,
            voxel_unit="nanometer",
        )

        # Compute actual disk size for the written layer
        if is_seg:
            layer_path = container_path / "labels" / (organelle if organelle else "segmentation")
        else:
            layer_path = container_path / "raw"
        size = sum(f.stat().st_size for f in layer_path.rglob("*") if f.is_file()) if layer_path.exists() else 0
        return size

    def _generate_pyramids(self, container_path: Path, has_seg: bool, organelle: str) -> None:
        """Generate multiscale pyramids for raw and segmentation layers."""
        from tensorswitch_v2.__main__ import find_base_level, run_local_pyramid

        # Raw pyramid
        raw_group = container_path / "raw"
        if raw_group.exists():
            try:
                s0_path, _ = find_base_level(str(raw_group))
                root_path = str(raw_group)
                print(f"    Generating raw pyramid from {s0_path}")
                run_local_pyramid(s0_path, root_path, downsample_method="mean", verbose=False)
            except Exception as e:
                print(f"    Raw pyramid generation failed: {e}")

        # Segmentation pyramid
        if has_seg:
            seg_key = organelle if organelle else "segmentation"
            seg_group = container_path / "labels" / seg_key
            if seg_group.exists():
                try:
                    s0_path, _ = find_base_level(str(seg_group))
                    root_path = str(seg_group)
                    print(f"    Generating segmentation pyramid from {s0_path}")
                    run_local_pyramid(s0_path, root_path, downsample_method="mode", verbose=False)
                except Exception as e:
                    print(f"    Segmentation pyramid generation failed: {e}")

    def run(self) -> DownloadReport:
        """Download subvolumes from all matching datasets within the size budget."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        budget_bytes = int(self.max_size_gb * 1024**3)

        num_datasets = len(self._datasets)
        budget_per_dataset = budget_bytes // num_datasets

        print(
            f"Downloading from {num_datasets} dataset(s), "
            f"~{budget_per_dataset / 1024**2:.0f} MB budget each "
            f"({self.max_size_gb:.1f} GB total)"
        )

        organelle_dir = self.organelle or "all"
        manifest_entries: list[DownloadedVolume] = []
        total_bytes = 0

        for i, entry in enumerate(self._datasets):
            backend = self._backends[entry.repository]
            ds_label = f"[{i + 1}/{num_datasets}] {entry.id}"

            try:
                scale, voxel_nm = self._pick_scale_and_voxel(entry, backend)
                vol_shape = backend.get_volume_shape(entry, scale)
            except Exception as e:
                print(f"  {ds_label}: skipping (metadata error: {e})")
                continue

            print(
                f"  {ds_label}: scale={scale}, "
                f"voxel={voxel_nm[0]:.1f}x{voxel_nm[1]:.1f}x{voxel_nm[2]:.1f} nm, "
                f"volume={vol_shape}"
            )

            # Probe raw dtype for budget calculation
            try:
                probe = backend.read_raw_crop(entry, (0, 0, 0), (1, 1, 1), scale)
                bytes_per_voxel = probe.dtype.itemsize
            except Exception as e:
                print(f"    skipping (read error: {e})")
                continue

            # Account for segmentation in budget (uint32 = 4 bytes)
            seg_bytes_per_voxel = 4 if self.require_segmentation else 0
            total_bpv = bytes_per_voxel + seg_bytes_per_voxel

            offset, shape = self._compute_subvolume(vol_shape, budget_per_dataset, total_bpv)
            print(f"    subvolume: offset={offset}, shape={shape}")

            out_dir = self.save_path / organelle_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            # Single OME-Zarr container for both raw and segmentation
            container_path = out_dir / f"{entry.id}.zarr"

            # Download raw
            try:
                raw_size = self._download_volume(
                    entry, backend, scale, offset, shape, voxel_nm,
                    container_path, label="raw",
                )
            except Exception as e:
                print(f"    raw download failed: {e}")
                continue

            # Download segmentation if requested
            seg_size = 0
            has_seg = False
            if self.require_segmentation and entry.has_segmentation:
                try:
                    seg_scale = backend.pick_seg_scale(entry, self.organelle, voxel_nm)
                    seg_voxel_nm = backend.get_seg_voxel_size(entry, self.organelle, seg_scale)
                    seg_size = self._download_volume(
                        entry, backend, seg_scale, offset, shape, seg_voxel_nm,
                        container_path, label="seg", organelle=self.organelle,
                    )
                    has_seg = True
                except Exception as e:
                    print(f"    seg download failed (may have different bounds): {e}")

            # Generate pyramids if requested
            has_pyramid = False
            if self.generate_pyramid:
                self._generate_pyramids(container_path, has_seg, self.organelle)
                has_pyramid = True

            vol_size = raw_size + seg_size
            total_bytes += vol_size

            # Write per-dataset metadata
            meta = {
                "dataset_id": entry.id,
                "repository": entry.repository,
                "organelle": self.organelle,
                "offset": list(offset),
                "shape": list(shape),
                "resolution_nm": list(voxel_nm),
                "scale_used": scale,
                "has_segmentation": has_seg,
                "has_pyramid": has_pyramid,
                "raw_source": entry.raw_path or entry.access_url,
            }
            meta_path = out_dir / f"{entry.id}_metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            manifest_entries.append(
                DownloadedVolume(
                    zarr_path=f"{organelle_dir}/{entry.id}.zarr",
                    dataset_id=entry.id,
                    repository=entry.repository,
                    organelle=self.organelle,
                    offset=list(offset),
                    shape=list(shape),
                    resolution_nm=list(voxel_nm),
                    scale_used=scale,
                    has_segmentation=has_seg,
                    has_pyramid=has_pyramid,
                    raw_dtype=str(probe.dtype),
                    size_bytes=vol_size,
                    raw_source=entry.raw_path or entry.access_url,
                )
            )

            print(f"    done ({vol_size / 1024**2:.1f} MB)")

        # Write manifest
        manifest_path = self.save_path / "manifest.json"
        manifest = {
            "version": 2,
            "created": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "organelle": self.organelle,
                "resolution_nm": list(self.resolution_nm) if self.resolution_nm else None,
                "max_size_gb": self.max_size_gb,
                "require_segmentation": self.require_segmentation,
                "modality_class": self.modality_class,
                "generate_pyramid": self.generate_pyramid,
            },
            "summary": {
                "num_volumes": len(manifest_entries),
                "num_datasets": len(set(v.dataset_id for v in manifest_entries)),
                "total_bytes": total_bytes,
            },
            "volumes": [asdict(v) for v in manifest_entries],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(
            f"\nComplete: {len(manifest_entries)} volume(s) from "
            f"{len(set(v.dataset_id for v in manifest_entries))} dataset(s), "
            f"{total_bytes / 1024**3:.2f} GB total"
        )
        print(f"Manifest: {manifest_path}")

        return DownloadReport(
            num_volumes=len(manifest_entries),
            num_datasets=len(set(v.dataset_id for v in manifest_entries)),
            total_bytes=total_bytes,
            manifest_path=str(manifest_path),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download contiguous subvolumes from microscopy repositories",
    )
    parser.add_argument("save_path", help="Directory to save downloaded data")
    parser.add_argument(
        "--organelle", default="", help="Filter by organelle (e.g. mito, er)",
    )
    parser.add_argument(
        "--resolution-nm",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Target resolution in nm (finest acceptable)",
    )
    parser.add_argument(
        "--max-size-gb", type=float, default=1.0, help="Approximate total size budget in GB",
    )
    parser.add_argument(
        "--repositories", nargs="*", default=None, help="Restrict to specific repositories",
    )
    parser.add_argument(
        "--require-segmentation", action="store_true", help="Only download datasets with segmentations",
    )
    parser.add_argument(
        "--modality-class", default="", help="Filter by modality (em, fluorescence)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed",
    )
    parser.add_argument(
        "--slab-size", type=int, default=64, help="Z-slices per read (controls memory usage)",
    )
    parser.add_argument(
        "--generate-pyramid", action="store_true", help="Generate multiscale pyramid after download",
    )

    args = parser.parse_args()

    downloader = DataDownloader(
        save_path=args.save_path,
        organelle=args.organelle,
        resolution_nm=tuple(args.resolution_nm) if args.resolution_nm else None,
        max_size_gb=args.max_size_gb,
        repositories=args.repositories,
        require_segmentation=args.require_segmentation,
        seed=args.seed,
        modality_class=args.modality_class,
        slab_size=args.slab_size,
        generate_pyramid=args.generate_pyramid,
    )
    report = downloader.run()
