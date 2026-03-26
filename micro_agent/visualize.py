"""Neuroglancer visualization for micro_agent crops and datasets.

Provides functions to display CropSamples (raw + segmentation) in an
interactive 3D viewer in the browser.

Usage:
    from micro_agent import UnifiedLoader
    from micro_agent.visualize import view_crop

    loader = UnifiedLoader(organelle="mito", crop_size=(64,64,64), scale=4)
    sample = next(iter(loader))
    viewer = view_crop(sample)
    # Navigate to the printed URL in your browser
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import neuroglancer
import numpy as np

if TYPE_CHECKING:
    from micro_agent.loader import CropSample
    from micro_agent.registry import DatasetEntry


def view_crop(
    sample: CropSample,
    voxel_size: tuple[float, float, float] | None = None,
) -> neuroglancer.Viewer:
    """Display a CropSample in neuroglancer with raw + segmentation overlay.

    Args:
        sample: A CropSample with .raw and .segmentation numpy arrays.
        voxel_size: (z, y, x) voxel size in nanometers.

    Returns:
        The neuroglancer Viewer instance. Keep a reference to prevent GC.
    """
    # Use resolution from the sample if available, otherwise fallback
    if voxel_size is None:
        voxel_size = sample.resolution_nm if any(v > 0 for v in sample.resolution_nm) else (8.0, 8.0, 8.0)

    viewer = neuroglancer.Viewer()

    coord_space = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        scales=list(voxel_size),
    )

    raw_volume = neuroglancer.LocalVolume(
        data=sample.raw,
        dimensions=coord_space,
        voxel_offset=list(sample.offset),
    )

    with viewer.txn() as s:
        s.layers["raw"] = neuroglancer.ImageLayer(source=raw_volume)

        if sample.segmentation is not None:
            seg_volume = neuroglancer.LocalVolume(
                data=sample.segmentation.astype(np.uint32),
                dimensions=coord_space,
                voxel_offset=list(sample.offset),
            )
            s.layers[sample.organelle + "_seg"] = neuroglancer.SegmentationLayer(
                source=seg_volume,
                selected_alpha=0.3,
            )

        center = [
            sample.offset[i] + sample.raw.shape[i] // 2
            for i in range(3)
        ]
        s.position = center

    print(f"Neuroglancer viewer: {viewer}")
    print(f"Dataset: {sample.dataset_id} ({sample.repository})")
    print(f"Organelle: {sample.organelle}")
    print(f"Offset: {sample.offset}, Shape: {sample.raw.shape}")
    print(f"Resolution: {voxel_size[0]:.1f}x{voxel_size[1]:.1f}x{voxel_size[2]:.1f} nm (source: {sample.source_resolution_nm[0]:.1f}x{sample.source_resolution_nm[1]:.1f}x{sample.source_resolution_nm[2]:.1f} nm @ s{sample.scale_used})")
    if sample.raw_path:
        print(f"Raw path: {sample.raw_path}")
    if sample.seg_path:
        print(f"Seg path: {sample.seg_path}")
    print(f"Segmentation: {sample.seg_status}")
    return viewer


def view_arrays(
    raw: np.ndarray,
    segmentation: np.ndarray | None = None,
    voxel_size: tuple[float, float, float] = (8.0, 8.0, 8.0),
    name: str = "volume",
) -> neuroglancer.Viewer:
    """Display raw numpy arrays in neuroglancer.

    Args:
        raw: 3D numpy array (z, y, x) for the EM image.
        segmentation: Optional 3D numpy array for segmentation overlay.
        voxel_size: (z, y, x) voxel size in nanometers.
        name: Name for the layer.

    Returns:
        The neuroglancer Viewer instance.
    """
    viewer = neuroglancer.Viewer()

    coord_space = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        scales=list(voxel_size),
    )

    raw_volume = neuroglancer.LocalVolume(data=raw, dimensions=coord_space)

    with viewer.txn() as s:
        s.layers[name] = neuroglancer.ImageLayer(source=raw_volume)

        if segmentation is not None:
            seg_volume = neuroglancer.LocalVolume(
                data=segmentation.astype(np.uint32),
                dimensions=coord_space,
            )
            s.layers[name + "_seg"] = neuroglancer.SegmentationLayer(
                source=seg_volume,
                selected_alpha=0.3,
            )

        center = [raw.shape[i] // 2 for i in range(3)]
        s.position = center

    print(f"Neuroglancer viewer: {viewer}")
    return viewer


def view_remote_n5(
    bucket: str,
    path: str,
    scale: int = 4,
    seg_path: str | None = None,
) -> neuroglancer.Viewer:
    """Display a remote N5 volume directly in neuroglancer via tensorstore.

    This shows the full dataset (not just a crop) using neuroglancer's
    built-in tensorstore integration.

    Args:
        bucket: S3 bucket name.
        path: Path to N5 array within bucket (e.g., "jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16").
        scale: Scale level to display.
        seg_path: Optional path to segmentation N5 array.

    Returns:
        The neuroglancer Viewer instance.
    """
    import tensorstore as ts

    viewer = neuroglancer.Viewer()

    em_spec = {
        "driver": "n5",
        "kvstore": {"driver": "s3", "bucket": bucket, "path": f"{path}/s{scale}"},
    }
    em_store = ts.open(em_spec, read=True).result()

    coord_space = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        scales=[8.0 * (2 ** scale)] * 3,
    )

    raw_volume = neuroglancer.LocalVolume(
        data=em_store,
        dimensions=coord_space,
    )

    with viewer.txn() as s:
        s.layers["em"] = neuroglancer.ImageLayer(source=raw_volume)

        if seg_path:
            seg_spec = {
                "driver": "n5",
                "kvstore": {"driver": "s3", "bucket": bucket, "path": f"{seg_path}/s{scale}"},
            }
            seg_store = ts.open(seg_spec, read=True).result()
            seg_volume = neuroglancer.LocalVolume(
                data=seg_store,
                dimensions=coord_space,
            )
            s.layers["segmentation"] = neuroglancer.SegmentationLayer(
                source=seg_volume,
                selected_alpha=0.3,
            )

        center = [em_store.shape[i] // 2 for i in range(3)]
        s.position = center

    print(f"Neuroglancer viewer: {viewer}")
    return viewer


# ---------------------------------------------------------------------------
# CLI entry point: pixi run view
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from micro_agent import UnifiedLoader

    organelle = sys.argv[1] if len(sys.argv) > 1 else "mito"
    crop_size = (128, 128, 128)
    resolution_nm = (8.0, 8.0, 8.0)

    print(f"Loading a {organelle} crop at {resolution_nm} nm...")
    loader = UnifiedLoader(
        organelle=organelle,
        crop_size=crop_size,
        resolution_nm=resolution_nm,
        num_samples=1,
        seed=42,
    )
    sample = next(iter(loader))

    print(f"Raw shape: {sample.raw.shape}, range: [{sample.raw.min()}, {sample.raw.max()}]")
    if sample.segmentation is not None:
        print(f"Seg shape: {sample.segmentation.shape}, unique: {np.unique(sample.segmentation)[:10]}")

    viewer = view_crop(sample)

    print("\nPress Ctrl+C to exit.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
