# Trailhead

Cross-repository microscopy training data discovery and loading. Trailhead provides a unified interface to query, load, and visualize EM datasets from multiple public repositories, yielding random crops at a target resolution suitable for training segmentation models.

## Repositories

| Repository | Format | Backend | Datasets |
|---|---|---|---|
| **OpenOrganelle** | Zarr / N5 on S3 | `OpenOrganelleBackend` | 51 (FIB-SEM, organelle segmentations) |
| **MICrONS** | Neuroglancer Precomputed on GCS | `MICrONSBackend` | 2 (minnie65, minnie35) |
| **FlyEM** | Neuroglancer Precomputed on GCS | `MICrONSBackend` | 4 (hemibrain, MANC, optic lobe, FIB-25) |
| **Google** | Neuroglancer Precomputed on GCS | `MICrONSBackend` | 4 (H01, FAFB, FlyWire, Kasthuri) |
| **OpenNeuroData** | Neuroglancer Precomputed on S3 | `MICrONSBackend` | 5 (Bock, Hildebrand, Harris, Wanner, Witvliet) |
| **EMPIAR** | TIFF slices over HTTPS | `EMPIARBackend` | 1 (on-demand slice download with local cache) |
| **IDR** | OME-Zarr on EBI S3 | `IDRBackend` | 1 |
| **CellMap Publications** | N5 on S3 | `OpenOrganelleBackend` | 1 (Heinrich 2021 ground truth crops) |

## Quick start

```bash
# Install with pixi
pixi install

# Search the registry
pixi run demo

# Open the web explorer
pixi run explore
# → http://ackermand-ws2:9000/

# View a single crop in neuroglancer
pixi run view
```

## Python API

### Load crops at a target resolution

```python
from trailhead import UnifiedLoader

loader = UnifiedLoader(
    organelle="mito",
    crop_size=(64, 64, 64),       # output size in voxels
    resolution_nm=(8.0, 8.0, 8.0), # target voxel size in nm
    num_samples=100,
    balance_repositories=True,     # equal sampling across repos
    require_nonempty_raw=True,     # skip crops with all-zero raw
    require_nonempty_seg=True,     # skip crops with no segmentation labels
)

print(loader.summary())

for sample in loader.prefetch_iter():
    print(f"{sample.dataset_id:30s} [{sample.repository}]")
    print(f"  resolution: {sample.resolution_nm} nm "
          f"(source: {sample.source_resolution_nm} nm @ s{sample.scale_used})")
    print(f"  raw: {sample.raw.shape}  seg: {sample.seg_status}")
    print(f"  path: {sample.raw_path}")
    # sample.raw       → np.ndarray (z,y,x) uint8
    # sample.segmentation → np.ndarray (z,y,x) uint32 or None
```

### Resolution-based scale selection

The loader automatically picks the best multiscale level for your target resolution and resamples to match:

- Reads voxel size metadata directly from volume files (neuroglancer precomputed `info` JSON, N5 `attributes.json`, zarr `.zattrs`)
- Picks the coarsest scale still finer than or equal to the target in all dimensions
- Computes per-axis zoom factors (`source_voxel / target_voxel`) and works backwards to determine how many source voxels to read: `read_shape = ceil(crop_size / zoom_factors)`. For example, requesting 64³ at 8nm from a 16nm source (zoom=2.0) reads 32³ source voxels, then resamples up to 64³.
- Resamples with `scipy.ndimage.zoom` (bilinear for raw, nearest-neighbor for segmentation) and trims/pads to exact `crop_size`
- Handles anisotropic data (e.g., MICrONS minnie65 at 8x8x40 nm gets per-axis zoom factors to produce isotropic output)

### Search the registry

```python
from trailhead import Registry

reg = Registry()

# Filter by organelle, organism, repository
hits = reg.search(organelle="mito", organism="Homo sapiens")
hits = reg.search(repository="FlyEM")
hits = reg.search(query="cortex", has_segmentation=True)

# List available metadata
reg.list_organelles()    # ['er', 'golgi', 'mito', 'neuron', ...]
reg.list_organisms()     # ['Caenorhabditis elegans', 'Drosophila melanogaster', ...]
reg.list_repositories()  # ['EMPIAR', 'FlyEM', 'Google', 'IDR', 'MICrONS', ...]
```

### Visualize in neuroglancer

```python
from trailhead import UnifiedLoader, view_crop

loader = UnifiedLoader(organelle="mito", crop_size=(64,64,64), resolution_nm=(8,8,8))
sample = next(iter(loader))
viewer = view_crop(sample)
# Opens neuroglancer in browser with raw + segmentation overlay
```

## Web explorer

`pixi run explore` starts a local web server with:

- **Control panel** -- organelle, resolution (nm), crop size, repository filters
- **Options** -- require segmentation, balance across repos, require nonzero raw/seg
- **Embedded neuroglancer** viewer with 1-99% intensity scaling and segment ID listing
- **Metadata bar** -- dataset ID, source/output resolution, offset, scale used, raw/seg paths, segmentation status
- **Background prefetching** -- keeps 5 crops ready in a queue for instant cycling
- **Keyboard shortcuts** -- Space or Right Arrow for next crop
- **Auto-cycle** -- automatically advance every N seconds

The server binds to `0.0.0.0` so it's accessible from other machines on the network.

## Project structure

```
trailhead/
  __init__.py              # Public API exports
  registry.py              # DatasetEntry + Registry (searchable catalog)
  loader.py                # UnifiedLoader (resolution-aware crop iterator)
  app.py                   # Web explorer (tornado + neuroglancer iframe)
  visualize.py             # Neuroglancer helpers (view_crop, view_arrays)
  discover.py              # Dataset auto-discovery agent
  backends/
    base.py                # Backend ABC (get_voxel_size, pick_scale, read crops)
    openorganelle.py       # Zarr-first + N5 fallback, multi-bucket S3
    microns.py             # Neuroglancer precomputed on GCS/S3 (HTTPS for GCS)
    empiar.py              # On-demand TIFF slice download via HTTPS
    idr.py                 # OME-Zarr on EBI S3
  catalog/
    openorganelle.json     # 51 datasets from s3://janelia-cosem-datasets
    microns.json           # MICrONS minnie65/35
```

## Pixi tasks

| Task | Command | Description |
|---|---|---|
| `demo` | `pixi run demo` | Search registry for mito datasets |
| `view` | `pixi run view` | Load one crop in neuroglancer |
| `explore` | `pixi run explore` | Web explorer with controls + viewer |
| `discover` | `pixi run discover` | Scan repos for new datasets |

## Dependencies

numpy, scipy, tensorstore, zarr, s3fs, httpx, neuroglancer
