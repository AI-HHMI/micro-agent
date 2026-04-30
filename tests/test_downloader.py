"""End-to-end downloader tests against a fake Backend.

The downloader places a *centered* subvolume sized to the per-dataset budget,
so given a known volume shape and budget we can assert the offset/shape exactly.
Tests run against a synthetic in-memory backend so they don't touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from micro_agent.backends.base import Backend
from micro_agent.downloader import DataDownloader
from micro_agent.registry import DatasetEntry


class FakeBackend(Backend):
    """In-memory backend with deterministic raw + segmentation arrays."""

    def __init__(
        self,
        volume_shape: tuple[int, int, int] = (32, 32, 32),
        voxel_nm: tuple[float, float, float] = (8.0, 8.0, 8.0),
        raw_dtype: np.dtype = np.dtype(np.uint8),
        seg_dtype: np.dtype = np.dtype(np.uint32),
    ) -> None:
        self.volume_shape = volume_shape
        self.voxel_nm = voxel_nm
        # Raw volume: ramp by z so any crop has predictable content.
        z = np.arange(volume_shape[0], dtype=raw_dtype).reshape(-1, 1, 1)
        self.raw = np.broadcast_to(z, volume_shape).astype(raw_dtype).copy()
        # Segmentation: single label = 7 inside a smaller centered cube, else 0
        self.seg = np.zeros(volume_shape, dtype=seg_dtype)
        cz, cy, cx = (s // 4 for s in volume_shape)
        self.seg[cz : 3 * cz, cy : 3 * cy, cx : 3 * cx] = 7

    def get_volume_shape(self, entry: DatasetEntry, scale: int = 0) -> tuple[int, ...]:
        return self.volume_shape

    def get_voxel_size(
        self, entry: DatasetEntry, scale: int = 0,
    ) -> tuple[float, float, float]:
        return self.voxel_nm

    def has_voxel_metadata(self, entry: DatasetEntry) -> bool:
        return True

    def pick_scale(
        self,
        entry: DatasetEntry,
        target_nm: tuple[float, float, float],
    ) -> int:
        return 0

    def read_raw_crop(
        self,
        entry: DatasetEntry,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> np.ndarray:
        oz, oy, ox = offset
        sz, sy, sx = shape
        return self.raw[oz : oz + sz, oy : oy + sy, ox : ox + sx].copy()

    def read_segmentation_crop(
        self,
        entry: DatasetEntry,
        organelle: str,
        offset: tuple[int, int, int],
        shape: tuple[int, int, int],
        scale: int = 0,
    ) -> np.ndarray:
        oz, oy, ox = offset
        sz, sy, sx = shape
        return self.seg[oz : oz + sz, oy : oy + sy, ox : ox + sx].copy()


def _make_entry(
    dataset_id: str,
    repository: str = "OpenOrganelle",
    has_segmentation: bool = True,
) -> DatasetEntry:
    return DatasetEntry(
        id=dataset_id,
        repository=repository,
        title=f"fake {dataset_id}",
        organelles=["mito"] if has_segmentation else [],
        has_segmentation=has_segmentation,
        supports_random_access=True,
        modality_class="em",
        access_url=f"file:///fake/{dataset_id}",
        raw_path=f"/fake/{dataset_id}",
    )


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend(volume_shape=(32, 32, 32), voxel_nm=(8.0, 8.0, 8.0))


@pytest.fixture
def patched_downloader(monkeypatch, fake_backend):
    """Build a DataDownloader with Registry + _get_backend patched."""

    def _build(
        entries: list[DatasetEntry],
        save_path: Path,
        **kwargs,
    ) -> DataDownloader:
        # Patch Registry so search() returns our fake entries.
        class FakeRegistry:
            def search(self, **_) -> list[DatasetEntry]:
                return list(entries)

            def list_organelles(self) -> list[str]:
                return ["mito"]

        monkeypatch.setattr("micro_agent.downloader.Registry", FakeRegistry)
        monkeypatch.setattr(
            "micro_agent.downloader._get_backend",
            lambda repo: fake_backend,
        )
        return DataDownloader(save_path=save_path, **kwargs)

    return _build


def _read_manifest(save_path: Path) -> dict:
    return json.loads((save_path / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_volume_fits_offset_is_zero(tmp_path, patched_downloader):
    """When the budget is huge, the entire volume is downloaded at offset (0,0,0)."""
    entry = _make_entry("ds_full")
    d = patched_downloader(
        [entry],
        tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=1.0,  # plenty
    )
    report = d.run()

    assert report.num_volumes == 1
    manifest = _read_manifest(tmp_path)
    vol = manifest["volumes"][0]
    assert vol["offset"] == [0, 0, 0]
    assert vol["shape"] == [32, 32, 32]
    assert vol["dataset_id"] == "ds_full"
    assert vol["resolution_nm"] == [8.0, 8.0, 8.0]


def test_centered_subvolume_when_budget_is_tight(tmp_path, patched_downloader):
    """Tight budget forces a smaller cube centered in the source volume."""
    backend = FakeBackend(volume_shape=(64, 64, 64))
    entry = _make_entry("ds_tight")

    # Budget: ~32^3 raw bytes + ~32^3 * 4 seg bytes ≈ 160KB ≈ 0.000149 GB
    # We want ~side=32 ⇒ centered offset = ((64-32)//2,) = 16
    d = patched_downloader(
        [entry],
        tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=160 * 1024 / 1024**3,  # ~160 KB
    )
    # Override backend so it has a 64³ source
    d._backends[entry.repository] = backend

    d.run()
    vol = _read_manifest(tmp_path)["volumes"][0]
    sz, sy, sx = vol["shape"]
    oz, oy, ox = vol["offset"]
    assert sz == sy == sx
    assert sz < 64
    # Centered placement: each axis offset = (64 - side) // 2
    expected_offset = (64 - sz) // 2
    assert (oz, oy, ox) == (expected_offset, expected_offset, expected_offset)


def test_writes_one_container_per_dataset(tmp_path, patched_downloader):
    """Multiple datasets produce one OME-Zarr container per id, under organelle/."""
    entries = [_make_entry(f"ds_{i}") for i in range(3)]
    d = patched_downloader(
        entries,
        tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=1.0,
    )
    d.run()

    organelle_dir = tmp_path / "mito"
    assert organelle_dir.is_dir()
    containers = sorted(p.name for p in organelle_dir.iterdir() if p.suffix == ".zarr")
    assert containers == ["ds_0.zarr", "ds_1.zarr", "ds_2.zarr"]
    for ds_id in ("ds_0", "ds_1", "ds_2"):
        container = organelle_dir / f"{ds_id}.zarr"
        assert (container / "raw" / "s0").is_dir()
        assert (container / "labels" / "mito" / "s0").is_dir()
        assert (organelle_dir / f"{ds_id}_metadata.json").is_file()


def test_raw_only_no_segmentation(tmp_path, patched_downloader):
    """With require_segmentation=False, only raw is written — no labels group."""
    entry = _make_entry("ds_raw", has_segmentation=False)
    d = patched_downloader(
        [entry],
        tmp_path,
        organelle="",
        require_segmentation=False,
        max_size_gb=1.0,
    )
    d.run()

    container = tmp_path / "all" / "ds_raw.zarr"
    assert (container / "raw" / "s0").is_dir()
    assert not (container / "labels").exists()


def test_manifest_records_per_dataset_offsets_and_shapes(tmp_path, patched_downloader):
    """The manifest captures one entry per downloaded volume with its placement."""
    entries = [_make_entry(f"ds_{i}") for i in range(2)]
    d = patched_downloader(
        entries,
        tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=1.0,
    )
    d.run()

    manifest = _read_manifest(tmp_path)
    assert manifest["summary"]["num_volumes"] == 2
    ids = sorted(v["dataset_id"] for v in manifest["volumes"])
    assert ids == ["ds_0", "ds_1"]
    for vol in manifest["volumes"]:
        assert vol["offset"] == [0, 0, 0]
        assert vol["shape"] == [32, 32, 32]
        assert vol["repository"] == "OpenOrganelle"


def test_roundtrip_raw_and_seg_match_fake_backend(tmp_path, patched_downloader, fake_backend):
    """Disk content equals what the FakeBackend served — proves writes are real."""
    import tensorstore as ts

    entry = _make_entry("ds_roundtrip")
    d = patched_downloader(
        [entry],
        tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=1.0,
    )
    d.run()

    container = tmp_path / "mito" / "ds_roundtrip.zarr"
    raw_array_path = container / "raw" / "s0"
    seg_array_path = container / "labels" / "mito" / "s0"

    raw_disk = ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(raw_array_path)},
    }).result().read().result()
    seg_disk = ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(seg_array_path)},
    }).result().read().result()

    assert raw_disk.shape == fake_backend.raw.shape
    assert seg_disk.shape == fake_backend.seg.shape
    assert np.array_equal(np.asarray(raw_disk), fake_backend.raw)
    assert np.array_equal(np.asarray(seg_disk), fake_backend.seg)

    # Verify sharding is enabled (PR #2's headline claim).
    raw_spec = json.loads((raw_array_path / "zarr.json").read_text())
    assert any(c["name"] == "sharding_indexed" for c in raw_spec["codecs"]), (
        "expected sharding_indexed codec — sharding regression"
    )


class OffCenterFakeBackend(FakeBackend):
    """Fake backend whose seg labels live in a corner — so a centered crop misses them."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seg = np.zeros_like(self.seg)
        sz, sy, sx = self.volume_shape
        # Put labels in the (0:sz/4, 0:sy/4, 0:sx/4) corner only.
        self.seg[: sz // 4, : sy // 4, : sx // 4] = 7


def _patch_registry_and_backend(monkeypatch, entries, backend):
    class FakeRegistry:
        def search(self, **_):
            return list(entries)

        def list_organelles(self):
            return ["mito"]

    monkeypatch.setattr("micro_agent.downloader.Registry", FakeRegistry)
    monkeypatch.setattr("micro_agent.downloader._get_backend", lambda repo: backend)


def test_random_placement_uses_seeded_rng(tmp_path, monkeypatch):
    """random_placement=True yields a non-centered, seed-reproducible offset."""
    backend = FakeBackend(volume_shape=(64, 64, 64))
    entry = _make_entry("ds_random")
    _patch_registry_and_backend(monkeypatch, [entry], backend)

    d = DataDownloader(
        save_path=tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=160 * 1024 / 1024**3,    # ~160KB → side ~32 (centered offset would be (16,16,16))
        random_placement=True,
        seed=42,
    )
    d._backends[entry.repository] = backend
    d.run()
    vol = _read_manifest(tmp_path)["volumes"][0]
    side = vol["shape"][0]
    centered = (64 - side) // 2
    assert vol["offset"] != [centered, centered, centered], (
        "random_placement should produce a non-centered offset"
    )
    # Reproducibility: same seed → same offset
    other_dir = tmp_path.parent / "rerun"
    DataDownloader(
        save_path=other_dir, organelle="mito", require_segmentation=True,
        max_size_gb=160 * 1024 / 1024**3, random_placement=True, seed=42,
    )._backends.update({entry.repository: backend})
    rerun = DataDownloader(
        save_path=other_dir, organelle="mito", require_segmentation=True,
        max_size_gb=160 * 1024 / 1024**3, random_placement=True, seed=42,
    )
    rerun._backends[entry.repository] = backend
    rerun.run()
    vol_rerun = _read_manifest(other_dir)["volumes"][0]
    assert vol["offset"] == vol_rerun["offset"]


def test_enforce_foreground_finds_off_center_labels(tmp_path, monkeypatch):
    """With OffCenterFakeBackend, centered placement returns all-zero seg;
    enforce_foreground must find the corner where labels live."""
    backend = OffCenterFakeBackend(volume_shape=(64, 64, 64))
    entry = _make_entry("ds_fg")
    _patch_registry_and_backend(monkeypatch, [entry], backend)

    # Sub_shape will be ~side=8 (tight budget). Labels live at [0:16, 0:16, 0:16].
    # Centered offset would be (28,28,28) — fully outside the labeled corner → all-zero.
    d = DataDownloader(
        save_path=tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=2560 / 1024**3,         # tiny, ~2.5KB → side ~8
        enforce_foreground=True,
        seed=0,
        max_placement_attempts=64,
    )
    d._backends[entry.repository] = backend
    d.run()

    vol = _read_manifest(tmp_path)["volumes"][0]
    sz = vol["shape"][0]
    oz, oy, ox = vol["offset"]
    # The chosen offset must overlap the labeled [0:16, 0:16, 0:16] corner.
    assert oz < 16 and oy < 16 and ox < 16, (
        f"enforce_foreground should land inside the labeled corner; got offset {vol['offset']}"
    )

    # And the on-disk seg actually contains nonzero labels.
    import tensorstore as ts
    seg_path = tmp_path / "mito" / "ds_fg.zarr" / "labels" / "mito" / "s0"
    seg = np.asarray(ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(seg_path)},
    }).result().read().result())
    assert np.any(seg != 0), "expected nonzero labels in saved seg"


def test_enforce_foreground_falls_back_when_nothing_works(tmp_path, monkeypatch):
    """When nothing in the volume has labels, fall back to centered placement
    (don't fail the run)."""
    backend = FakeBackend(volume_shape=(32, 32, 32))
    backend.seg = np.zeros_like(backend.seg)        # zero out — no labels anywhere
    entry = _make_entry("ds_no_fg")
    _patch_registry_and_backend(monkeypatch, [entry], backend)

    d = DataDownloader(
        save_path=tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=1.0,
        enforce_foreground=True,
        seed=0,
        max_placement_attempts=4,
    )
    d._backends[entry.repository] = backend
    d.run()

    vol = _read_manifest(tmp_path)["volumes"][0]
    # With shape == volume_shape (full vol fits), centered offset is (0,0,0).
    assert vol["offset"] == [0, 0, 0]


def test_different_repositories_use_per_repo_backend_lookup(
    tmp_path, monkeypatch, fake_backend,
):
    """One DataDownloader can serve datasets from different repositories."""
    entries = [
        _make_entry("ds_oo", repository="OpenOrganelle"),
        _make_entry("ds_microns", repository="MICrONS"),
    ]

    class FakeRegistry:
        def search(self, **_) -> list[DatasetEntry]:
            return list(entries)

        def list_organelles(self) -> list[str]:
            return ["mito"]

    seen_repos: list[str] = []

    def fake_get_backend(repo: str) -> Backend:
        seen_repos.append(repo)
        return fake_backend

    monkeypatch.setattr("micro_agent.downloader.Registry", FakeRegistry)
    monkeypatch.setattr("micro_agent.downloader._get_backend", fake_get_backend)

    DataDownloader(
        save_path=tmp_path,
        organelle="mito",
        require_segmentation=True,
        max_size_gb=1.0,
    ).run()

    # _get_backend is called once per unique repository encountered.
    assert sorted(set(seen_repos)) == ["MICrONS", "OpenOrganelle"]
    for ds_id in ("ds_oo", "ds_microns"):
        assert (tmp_path / "mito" / f"{ds_id}.zarr" / "raw" / "s0").is_dir()
