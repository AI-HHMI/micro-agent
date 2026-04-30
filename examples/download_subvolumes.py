"""Interactive example: download subvolumes and read them back.

Run cell-by-cell in VS Code (Python extension) or Jupyter — the `# %%`
markers are recognized as cell delimiters. Or run top-to-bottom:

    pixi run python examples/download_subvolumes.py
"""

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
import json
import random

import numpy as np

from micro_agent.registry import Registry, DatasetEntry
from micro_agent.downloader import DataDownloader
from micro_agent.backends import TensorSwitchBackend

SAVE_PATH = Path("./training_data")
SAVE_PATH.mkdir(exist_ok=True)


# %% [markdown]
# ## Browse what's available
#
# Inspect the registry before downloading — what organelles, organisms,
# and repositories are catalogued.

# %%
registry = Registry()
print(f"Registry has {len(registry)} datasets")
print(f"Organelles: {registry.list_organelles()[:10]} ...")
print(f"Repositories: {registry.list_repositories()}")
mito_hits = registry.search(organelle="mito", has_segmentation=True)
print(f"\n{len(mito_hits)} datasets have mito segmentations:")
for entry in mito_hits[:5]:
    print(f"  {entry.id:30s} ({entry.repository})")


# %% [markdown]
# ## Example 1 — equal byte budget per dataset (default behavior)
#
# `DataDownloader` divides `max_size_gb` evenly across every matching
# dataset and takes one centered subvolume from each. Bias warning: if
# OpenOrganelle has 50 mito datasets and IDR has 5, OpenOrganelle ends
# up with 10× more bytes on disk.

# %%
DataDownloader(
    save_path=SAVE_PATH / "ex1_equal_per_dataset",
    organelle="mito",
    resolution_nm=(8.0, 8.0, 8.0),
    # Per-dataset side ≈ cbrt(max_size_gb*1024**3 / N_datasets / 5).
    # Tensorswitch only builds pyramids when sides exceed the inner chunk
    # shape (default 64³), so we need ≳130³ subvolumes for s1 to appear.
    # 0.5 GB / ~13 datasets / 5 B/voxel → ~196³ → s1 = 98³ (pyramidable).
    max_size_gb=0.5,
    require_segmentation=True,
    repositories=["OpenOrganelle"],
    modality_class="em",
    generate_pyramid=True,
).run()


# %% [markdown]
# ## Example 2 — equal byte budget per *source*
#
# Pattern: discover which sources actually have matches, then loop
# `DataDownloader` once per source with that source's slice of the
# total budget.

# %%
TOTAL_GB = 0.1
sources_with_mito = sorted({h.repository for h in mito_hits})
per_source_gb = TOTAL_GB / len(sources_with_mito)
print(f"Splitting {TOTAL_GB} GB across {len(sources_with_mito)} sources "
      f"({per_source_gb:.3f} GB each)")

for source in sources_with_mito:
    print(f"\n=== {source} ===")
    DataDownloader(
        save_path=SAVE_PATH / "ex2_equal_per_source",
        organelle="mito",
        resolution_nm=(8.0, 8.0, 8.0),
        max_size_gb=per_source_gb,
        require_segmentation=True,
        repositories=[source],
        modality_class="em",
    ).run()


# %% [markdown]
# ## Example 3 — random subset of datasets
#
# By default the downloader takes one centered crop from *every*
# matching dataset. To instead grab N random datasets, restrict the
# selection ahead of time by passing only those entries' ids via the
# repositories filter — or hack the internal datasets list.
#
# Reproducible random pick:

# %%
rng = random.Random(42)
N_PICK = 3
chosen = rng.sample(mito_hits, k=min(N_PICK, len(mito_hits)))
print(f"Randomly picked {len(chosen)} datasets:")
for e in chosen:
    print(f"  {e.id} ({e.repository})")

d = DataDownloader(
    save_path=SAVE_PATH / "ex3_random_subset",
    organelle="mito",
    resolution_nm=(8.0, 8.0, 8.0),
    max_size_gb=0.05,
    require_segmentation=True,
    modality_class="em",
)
# Override the auto-discovered list with our random pick.
d._datasets = chosen
# Re-resolve backends for any new repositories the random pick brought in.
from micro_agent.loader import _get_backend
for entry in chosen:
    d._backends.setdefault(entry.repository, _get_backend(entry.repository))
d.run()


# %% [markdown]
# ## Example 4 — enforce foreground (only download crops with labels)
#
# Default placement is centered, which can land in regions with no
# segmentation labels (especially for sparse organelles like mito). Use
# `enforce_foreground=True` to probe random offsets and reject all-zero
# crops. Falls back to centered if no labeled location is found within
# `max_placement_attempts`.

# %%
DataDownloader(
    save_path=SAVE_PATH / "ex4_foreground",
    organelle="mito",
    resolution_nm=(8.0, 8.0, 8.0),
    max_size_gb=0.5,
    require_segmentation=True,
    repositories=["OpenOrganelle"],
    modality_class="em",
    enforce_foreground=True,         # ← reject all-zero seg crops
    seed=42,                          # reproducible random offsets
    max_placement_attempts=16,
).run()


# %% [markdown]
# ## Example 5 — random placement (variety, not centered)
#
# Use `random_placement=True` for diverse training crops without the
# foreground guarantee. With a seed, runs are reproducible.

# %%
DataDownloader(
    save_path=SAVE_PATH / "ex5_random",
    organelle="mito",
    resolution_nm=(8.0, 8.0, 8.0),
    max_size_gb=0.2,
    require_segmentation=True,
    repositories=["OpenOrganelle"],
    modality_class="em",
    random_placement=True,
    seed=42,
).run()


# %% [markdown]
# ## Read back via TensorSwitchBackend
#
# The downloader's output is itself a valid micro-agent backend, so you
# can plug it into the loader / training pipeline like any other source.

# %%
backend = TensorSwitchBackend()

for container in sorted((SAVE_PATH / "ex1_equal_per_dataset" / "mito").glob("*.zarr")):
    entry = DatasetEntry(
        id=container.stem,
        repository="TensorSwitch",
        title=container.stem,
        raw_path=str(container),
    )
    print(f"\n{container.name}")
    print(f"  shape s0 = {backend.get_volume_shape(entry, scale=0)}")
    print(f"  voxel s0 = {backend.get_voxel_size(entry, scale=0)} nm")
    print(f"  pick_scale @ 16 nm = s{backend.pick_scale(entry, (16.0, 16.0, 16.0))}")

    raw = backend.read_raw_crop(entry, offset=(0, 0, 0), shape=(32, 32, 32))
    print(f"  raw {raw.shape} {raw.dtype}, range [{raw.min()}, {raw.max()}]")

    # Segmentation isn't always present — some datasets' seg has different
    # bounds than raw and the downloader skips it. Check before reading.
    if (container / "labels" / "mito").exists():
        seg = backend.read_segmentation_crop(entry, "mito", offset=(0, 0, 0), shape=(32, 32, 32))
        print(f"  seg {seg.shape} {seg.dtype}, unique labels: {len(np.unique(seg))}")
    else:
        print("  seg: not downloaded (likely raw/seg bounds mismatch — see manifest.json)")


# %% [markdown]
# ## Direct tensorstore read (bypassing the backend)

# %%
import tensorstore as ts

first = next((SAVE_PATH / "ex1_equal_per_dataset" / "mito").glob("*.zarr"))
arr = ts.open({
    "driver": "zarr3",
    "kvstore": {"driver": "file", "path": str(first / "raw" / "s0")},
}).result()
print(f"shape={arr.shape}, dtype={arr.dtype}")

# Verify sharding is on (PR #2's headline change).
spec = json.loads((first / "raw" / "s0" / "zarr.json").read_text())
print(f"codecs: {[c['name'] for c in spec['codecs']]}")
assert any(c["name"] == "sharding_indexed" for c in spec["codecs"])
print("✓ shard codec present")
