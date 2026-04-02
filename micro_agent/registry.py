"""Dataset registry: searchable catalog of datasets across microscopy repositories.

The registry knows what datasets exist, what organisms/organelles they contain,
what formats they're in, and how to access them. It combines a curated local
catalog with dynamically discovered entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetEntry:
    """A single dataset in the registry."""

    id: str
    repository: str
    title: str
    organism: str = ""
    cell_type: str = ""
    imaging_modality: str = ""
    voxel_size_nm: list[float] = field(default_factory=list)
    organelles: list[str] = field(default_factory=list)
    has_raw: bool = True
    has_segmentation: bool = False
    data_format: str = ""
    access_url: str = ""
    raw_path: str = ""
    segmentation_paths: dict[str, str] = field(default_factory=dict)
    supports_random_access: bool = True  # False for FTP/download-only repos like EMPIAR
    # Multi-channel / fluorescence fields
    num_channels: int = 0  # 0 = unknown
    channel_names: list[str] = field(default_factory=list)  # e.g. ["DAPI", "GFP", "mCherry"]
    wavelengths_nm: list[float] = field(default_factory=list)  # e.g. [405, 488, 561]
    fluorophores: list[str] = field(default_factory=list)
    modality_class: str = ""  # "em" | "fluorescence" | "correlative"
    validation_status: str = "pending"  # "verified" | "failed" | "pending"

    def matches(self, query: str) -> bool:
        """Check if this entry matches a free-text query."""
        q = query.lower()
        searchable = " ".join(
            [
                self.id,
                self.title,
                self.organism,
                self.cell_type,
                self.imaging_modality,
                " ".join(self.organelles),
                " ".join(self.channel_names),
                " ".join(self.fluorophores),
            ]
        ).lower()
        return all(term in searchable for term in q.split())


# ---------------------------------------------------------------------------
# Catalog loading helpers
# ---------------------------------------------------------------------------

CATALOG_DIR = Path(__file__).parent / "catalog"


def _load_openorganelle_catalog() -> list[DatasetEntry]:
    """Load OpenOrganelle datasets from the JSON catalog."""
    catalog_path = CATALOG_DIR / "openorganelle.json"
    if not catalog_path.exists():
        return []

    with open(catalog_path) as f:
        items = json.load(f)

    entries: list[DatasetEntry] = []
    for item in items:
        ds_id = item["id"]
        em_name = item.get("em_name", "fibsem-uint16")
        organelles = item.get("organelles", [])
        # Strip _seg suffix from organelle names — the catalog stores bare names
        clean_organelles = [o.replace("_seg", "") for o in organelles]

        # Use explicit raw_path/data_format if provided, otherwise default to N5
        raw_path = item.get("raw_path", f"{ds_id}/{ds_id}.n5/em/{em_name}")
        data_format = item.get("data_format", "n5")
        seg_paths = item.get("segmentation_paths")
        if seg_paths is None:
            n5_base = f"{ds_id}/{ds_id}.n5"
            seg_paths = {o: f"{n5_base}/labels/{o}_seg" for o in clean_organelles}

        entries.append(DatasetEntry(
            id=ds_id,
            repository="OpenOrganelle",
            title=item.get("title", ds_id),
            organism=item.get("organism", ""),
            cell_type=item.get("cell_type", ""),
            imaging_modality=item.get("imaging_modality", "FIB-SEM"),
            voxel_size_nm=item.get("voxel_size_nm", [8.0, 8.0, 8.0]),
            organelles=clean_organelles,
            has_segmentation=item.get("has_segmentation", len(organelles) > 0),
            has_raw=True,
            data_format=data_format,
            access_url=f"s3://janelia-cosem-datasets/{ds_id}/",
            raw_path=raw_path,
            segmentation_paths=seg_paths,
            modality_class="em",
        ))
    return entries


def _load_microns_catalog() -> list[DatasetEntry]:
    """Load MICrONS datasets from the JSON catalog."""
    catalog_path = CATALOG_DIR / "microns.json"
    if not catalog_path.exists():
        return []

    with open(catalog_path) as f:
        items = json.load(f)

    entries: list[DatasetEntry] = []
    for item in items:
        entries.append(DatasetEntry(
            id=item["id"],
            repository="OpenNeuroData",
            title=item.get("title", item["id"]),
            organism=item.get("organism", ""),
            cell_type=item.get("cell_type", ""),
            imaging_modality="FIB-SEM",
            organelles=item.get("organelles", []),
            has_segmentation=item.get("has_segmentation", False),
            has_raw=True,
            data_format="neuroglancer_precomputed",
            raw_path=item.get("raw_url", ""),
            segmentation_paths={
                org: item.get("seg_url", "") for org in item.get("organelles", [])
            },
            modality_class="em",
        ))
    return entries


# Stub entries for IDR and EMPIAR — these repos require live API queries
# for full catalog, but we include a few known entries.
_EMPIAR_ENTRIES = [
    DatasetEntry(
        id="EMPIAR-10310",
        repository="EMPIAR",
        title="FIB-SEM of Platynereis parapodia (Müller et al.)",
        organism="Platynereis dumerilii",
        cell_type="parapodia",
        imaging_modality="FIB-SEM",
        has_segmentation=False,
        has_raw=True,
        data_format="tiff",
        access_url="https://ftp.ebi.ac.uk/empiar/world_availability/10310/data/",
        raw_path="20180813_platynereis_parapodia/raw_16bit",
        modality_class="em",
    ),
]

_IDR_ENTRIES = [
    DatasetEntry(
        id="9836842",
        repository="IDR",
        title="3D FIB-SEM of yeast cells (Xu et al.)",
        organism="Saccharomyces cerevisiae",
        cell_type="yeast",
        imaging_modality="FIB-SEM",
        has_segmentation=False,
        has_raw=True,
        data_format="ome-zarr",
        supports_random_access=False,  # EBI S3 not reachable from Janelia
        modality_class="em",
    ),
]

# GCS-hosted datasets using neuroglancer_precomputed format
# These all use the MICrONS backend (same tensorstore driver)
_GCS_ENTRIES = [
    DatasetEntry(
        id="h01_human_cortex",
        repository="Google",
        title="H01 — Human cortex petascale reconstruction (Google/Lichtman)",
        organism="Homo sapiens",
        cell_type="temporal cortex neuron",
        imaging_modality="serial-section EM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://h01-release/data/20210601/4nm_raw",
        segmentation_paths={"neuron": "gs://h01-release/data/20210601/c3"},
        modality_class="em",
    ),
    DatasetEntry(
        id="hemibrain_v1.2",
        repository="FlyEM",
        title="FlyEM hemibrain v1.2 — Drosophila central brain connectome",
        organism="Drosophila melanogaster",
        cell_type="central brain neuron",
        imaging_modality="FIB-SEM",
        organelles=["neuron", "mito"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg",
        segmentation_paths={
            "neuron": "gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
            "mito": "gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects",
        },
        modality_class="em",
    ),
    DatasetEntry(
        id="fafb_v14",
        repository="Google",
        title="FAFB v14 — Full adult fly brain EM",
        organism="Drosophila melanogaster",
        cell_type="whole brain neuron",
        imaging_modality="serial-section TEM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig",
        segmentation_paths={"neuron": "gs://fafb-ffn1-20200412/segmentation"},
        modality_class="em",
    ),
    DatasetEntry(
        id="kasthuri2011",
        repository="Google",
        title="Kasthuri 2011 — Mouse somatosensory cortex (saturated reconstruction)",
        organism="Mus musculus",
        cell_type="somatosensory cortex",
        imaging_modality="serial-section EM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://neuroglancer-public-data/kasthuri2011/image",
        segmentation_paths={"neuron": "gs://neuroglancer-public-data/kasthuri2011/ground_truth"},
        modality_class="em",
    ),
    DatasetEntry(
        id="flyem_fib25",
        repository="FlyEM",
        title="FlyEM FIB-25 — Drosophila medulla 7-column",
        organism="Drosophila melanogaster",
        cell_type="medulla neuron",
        imaging_modality="FIB-SEM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://neuroglancer-public-data/flyem_fib-25/image",
        segmentation_paths={"neuron": "gs://neuroglancer-public-data/flyem_fib-25/ground_truth"},
        modality_class="em",
    ),
    # Janelia FlyEM — MANC (Male Adult Nerve Cord)
    DatasetEntry(
        id="manc_v1.0",
        repository="FlyEM",
        title="MANC — Male adult nerve cord connectome (Janelia FlyEM)",
        organism="Drosophila melanogaster",
        cell_type="VNC + brain neuron",
        imaging_modality="FIB-SEM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://flyem-male-cns/em/em-clahe-jpeg",
        segmentation_paths={"neuron": "gs://flyem-male-cns/v0.9/segmentation"},
        modality_class="em",
    ),
    # Janelia FlyEM — Optic Lobe (not publicly accessible yet)
    DatasetEntry(
        id="flyem_optic_lobe",
        repository="FlyEM",
        title="FlyEM optic lobe — Drosophila visual system connectome",
        organism="Drosophila melanogaster",
        cell_type="optic lobe neuron",
        imaging_modality="FIB-SEM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://flyem-optic-lobe/em/em-clahe-jpeg",
        segmentation_paths={"neuron": "gs://flyem-optic-lobe/v1.1/segmentation"},
        supports_random_access=False,
        modality_class="em",
    ),
    # FlyWire (FAFB with Seung Lab segmentation)
    DatasetEntry(
        id="flywire_fafb",
        repository="Google",
        title="FlyWire — Full adult fly brain (Seung Lab proofread connectome)",
        organism="Drosophila melanogaster",
        cell_type="whole brain neuron",
        imaging_modality="serial-section TEM",
        organelles=["neuron"],
        has_segmentation=False,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="gs://microns-seunglab/drosophila_v0/alignment/image_rechunked",
        modality_class="em",
    ),
]

# Open NeuroData datasets on AWS S3 (neuroglancer_precomputed)
_OPENNEURODATA_ENTRIES = [
    DatasetEntry(
        id="bock11",
        repository="OpenNeuroData",
        title="Bock et al. 2011 — Mouse primary visual cortex",
        organism="Mus musculus",
        cell_type="visual cortex",
        imaging_modality="serial-section TEM",
        has_segmentation=False,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="s3://open-neurodata/bock11/image",
        modality_class="em",
    ),
    DatasetEntry(
        id="hildebrand_zebrafish",
        repository="OpenNeuroData",
        title="Hildebrand 2017 — Larval zebrafish whole-brain EM",
        organism="Danio rerio",
        cell_type="whole brain",
        imaging_modality="serial-section TEM",
        has_segmentation=False,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="s3://open-neurodata/hildebrand/130201zf142/160515_SWiFT_60nmpx",
        modality_class="em",
    ),
    DatasetEntry(
        id="kharris15_spine",
        repository="OpenNeuroData",
        title="Harris 2015 — Rat hippocampal CA1 neuropil",
        organism="Rattus norvegicus",
        cell_type="hippocampal CA1",
        imaging_modality="serial-section TEM",
        organelles=["neuron"],
        has_segmentation=True,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="s3://open-neurodata/kharris15/spine/em",
        segmentation_paths={"neuron": "s3://open-neurodata/kharris15/spine/anno"},
        modality_class="em",
    ),
    DatasetEntry(
        id="wanner16_zebrafish_ob",
        repository="OpenNeuroData",
        title="Wanner 2016 — Zebrafish olfactory bulb (1022 neurons)",
        organism="Danio rerio",
        cell_type="olfactory bulb",
        imaging_modality="SBF-SEM",
        has_segmentation=False,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="s3://open-neurodata/wanner16/AA201605/SBEM1",
        modality_class="em",
    ),
    DatasetEntry(
        id="witvliet2020_celegans_1",
        repository="OpenNeuroData",
        title="Witvliet 2020 — C. elegans developmental connectome (stage 1)",
        organism="Caenorhabditis elegans",
        cell_type="whole brain",
        imaging_modality="serial-section TEM",
        has_segmentation=False,
        has_raw=True,
        data_format="neuroglancer_precomputed",
        raw_path="s3://bossdb-open-data/witvliet2020/Dataset_1/em",
        modality_class="em",
    ),
]

# CellMap publications bucket — ground truth training crops
_CELLMAP_PUBLICATIONS_ENTRIES = [
    DatasetEntry(
        id="cellmap_gt_hela2",
        repository="OpenOrganelle",
        title="CellMap ground truth — HeLa-2 training crops (Heinrich 2021)",
        organism="Homo sapiens",
        cell_type="HeLa",
        imaging_modality="FIB-SEM",
        organelles=["mito", "er", "nucleus", "golgi", "vesicle", "pm", "endo"],
        has_segmentation=True,
        has_raw=True,
        data_format="n5",
        access_url="s3://janelia-cosem-publications/",
        raw_path="heinrich-2021a/jrc_hela-2/jrc_hela-2.n5/volumes/raw",
        segmentation_paths={
            org: f"heinrich-2021a/jrc_hela-2/jrc_hela-2.n5/volumes/labels/{org}" for org in
            ["mito", "er", "nucleus", "golgi", "vesicle", "pm", "endo"]
        },
        modality_class="em",
    ),
]


class Registry:
    """Searchable catalog of microscopy datasets across repositories.

    Usage:
        registry = Registry()
        hits = registry.search("mitochondria")
        hits = registry.search("mito", organism="Homo sapiens")
        hits = registry.search("er", repository="OpenOrganelle")
    """

    def __init__(self, load_discovered: bool = True) -> None:
        self._entries: list[DatasetEntry] = []
        self._load_curated()
        if load_discovered:
            self._auto_load_discovered()

    def _load_curated(self) -> None:
        """Load all curated catalogs."""
        self._entries.extend(_load_openorganelle_catalog())
        self._entries.extend(_load_microns_catalog())
        self._entries.extend(_EMPIAR_ENTRIES)
        self._entries.extend(_IDR_ENTRIES)
        self._entries.extend(_GCS_ENTRIES)
        self._entries.extend(_OPENNEURODATA_ENTRIES)
        self._entries.extend(_CELLMAP_PUBLICATIONS_ENTRIES)

    def _auto_load_discovered(self) -> None:
        """Auto-load discovered datasets from common locations."""
        # Check for discovered_datasets.json in the project root and working dir
        candidates = [
            Path(__file__).parent.parent / "discovered_datasets.json",
            Path("discovered_datasets.json"),
        ]
        existing_ids = {e.id for e in self._entries}
        for path in candidates:
            if path.exists():
                count = self.load_discovered(path, skip_ids=existing_ids)
                if count > 0:
                    break

    def add(self, entry: DatasetEntry) -> None:
        """Add a dataset entry to the registry."""
        self._entries.append(entry)

    def load_discovered(self, path: str | Path, skip_ids: set[str] | None = None) -> int:
        """Load entries from a discovery JSON file. Returns count added."""
        path = Path(path)
        if not path.exists():
            return 0
        with open(path) as f:
            items = json.load(f)
        count = 0
        for item in items:
            if skip_ids and item.get("id", "") in skip_ids:
                continue
            entry = DatasetEntry(
                id=item["id"],
                repository=item.get("repository", "discovered"),
                title=item.get("title", item["id"]),
                organism=item.get("organism", ""),
                cell_type=item.get("cell_type", ""),
                imaging_modality=item.get("imaging_modality", ""),
                organelles=item.get("organelles", []),
                has_segmentation=item.get("has_segmentation", False),
                has_raw=item.get("has_raw", True),
                data_format=item.get("data_format", ""),
                access_url=item.get("access_url", ""),
                raw_path=item.get("raw_path", ""),
                segmentation_paths=item.get("segmentation_paths", {}),
                supports_random_access=item.get("supports_random_access", True),
                num_channels=item.get("num_channels", 0),
                channel_names=item.get("channel_names", []),
                wavelengths_nm=item.get("wavelengths_nm", []),
                fluorophores=item.get("fluorophores", []),
                modality_class=item.get("modality_class", ""),
                validation_status=item.get("validation_status", "pending"),
            )
            self._entries.append(entry)
            count += 1
        return count

    @property
    def entries(self) -> list[DatasetEntry]:
        return list(self._entries)

    def search(
        self,
        query: str = "",
        *,
        organelle: str = "",
        organism: str = "",
        cell_type: str = "",
        repository: str = "",
        has_segmentation: bool | None = None,
        modality_class: str = "",
        min_channels: int = 0,
        validation_status: str = "",
    ) -> list[DatasetEntry]:
        """Search the registry with optional filters."""
        results = self._entries

        if query:
            results = [e for e in results if e.matches(query)]

        if organelle:
            organelle_lower = organelle.lower()
            results = [
                e for e in results if any(organelle_lower in o.lower() for o in e.organelles)
            ]

        if organism:
            organism_lower = organism.lower()
            results = [e for e in results if organism_lower in e.organism.lower()]

        if cell_type:
            cell_type_lower = cell_type.lower()
            results = [e for e in results if cell_type_lower in e.cell_type.lower()]

        if repository:
            repo_lower = repository.lower()
            results = [e for e in results if repo_lower in e.repository.lower()]

        if has_segmentation is not None:
            results = [e for e in results if e.has_segmentation == has_segmentation]

        if modality_class:
            mc_lower = modality_class.lower()
            results = [e for e in results if e.modality_class.lower() == mc_lower]

        if min_channels > 0:
            results = [e for e in results if e.num_channels >= min_channels]

        if validation_status:
            vs_lower = validation_status.lower()
            results = [e for e in results if e.validation_status.lower() == vs_lower]

        return results

    def list_organelles(self) -> list[str]:
        """Return all unique organelle names in the registry."""
        organelles: set[str] = set()
        for entry in self._entries:
            organelles.update(entry.organelles)
        return sorted(organelles)

    def list_organisms(self) -> list[str]:
        """Return all unique organism names in the registry."""
        return sorted({e.organism for e in self._entries if e.organism})

    def list_repositories(self) -> list[str]:
        """Return all unique repository names in the registry."""
        return sorted({e.repository for e in self._entries})

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({len(self._entries)} datasets)"
