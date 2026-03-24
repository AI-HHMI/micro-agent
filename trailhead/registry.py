"""Dataset registry: searchable catalog of datasets across microscopy repositories.

The registry knows what datasets exist, what organisms/organelles they contain,
what formats they're in, and how to access them. It combines a curated local
catalog with live API queries where available.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
            ]
        ).lower()
        return all(term in searchable for term in q.split())


# ---------------------------------------------------------------------------
# Curated catalog — these are datasets we know have both EM + segmentations
# and are accessible on public cloud storage.
# ---------------------------------------------------------------------------

def _oo_entry(
    id: str,
    title: str,
    organism: str,
    cell_type: str,
    organelles: list[str],
    em_name: str = "fibsem-uint16",
    voxel_size_nm: list[float] | None = None,
) -> DatasetEntry:
    """Helper to create an OpenOrganelle dataset entry with standard paths.

    S3 layout: {id}/{id}.n5/em/{em_name}/s{N} for raw
               {id}/{id}.n5/labels/{organelle}_seg/s{N} for segmentations
    """
    n5_base = f"{id}/{id}.n5"
    return DatasetEntry(
        id=id,
        repository="OpenOrganelle",
        title=title,
        organism=organism,
        cell_type=cell_type,
        imaging_modality="FIB-SEM",
        voxel_size_nm=voxel_size_nm or [8.0, 8.0, 8.0],
        organelles=organelles,
        has_segmentation=True,
        data_format="n5",
        access_url=f"s3://janelia-cosem-datasets/{id}/",
        raw_path=f"{n5_base}/em/{em_name}",
        segmentation_paths={org: f"{n5_base}/labels/{org}_seg" for org in organelles},
    )


# Verified against actual S3 bucket contents (2026-03-24).
_OPENORGANELLE_DATASETS = [
    _oo_entry("jrc_hela-2", "HeLa cell (Interphase) #2",
              "Homo sapiens", "HeLa",
              ["mito", "er", "nucleus", "golgi", "vesicle", "mt-out", "pm", "endo"],
              em_name="fibsem-uint16"),
    _oo_entry("jrc_hela-3", "HeLa cell (Interphase) #3",
              "Homo sapiens", "HeLa",
              ["mito", "er", "nucleus", "golgi"],
              em_name="fibsem-uint16",
              voxel_size_nm=[4.0, 4.0, 4.0]),
    _oo_entry("jrc_macrophage-2", "Macrophage #2",
              "Homo sapiens", "macrophage",
              ["mito", "er", "nucleus"],
              em_name="fibsem-uint16"),
    _oo_entry("jrc_mus-liver", "Mouse liver",
              "Mus musculus", "hepatocyte",
              ["mito", "er", "nucleus", "pm"],
              em_name="fibsem-uint8"),
]


class Registry:
    """Searchable catalog of microscopy datasets across repositories.

    Usage:
        registry = Registry()
        hits = registry.search("mitochondria")
        hits = registry.search("mito", organism="Homo sapiens")
        hits = registry.search("er", repository="OpenOrganelle")
    """

    def __init__(self) -> None:
        self._entries: list[DatasetEntry] = []
        self._load_curated()

    def _load_curated(self) -> None:
        """Load the built-in curated catalog."""
        self._entries.extend(_OPENORGANELLE_DATASETS)

    def add(self, entry: DatasetEntry) -> None:
        """Add a dataset entry to the registry."""
        self._entries.append(entry)

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
    ) -> list[DatasetEntry]:
        """Search the registry with optional filters.

        Args:
            query: Free-text search across all fields.
            organelle: Filter by organelle name (e.g., "mito", "er").
            organism: Filter by organism (e.g., "Homo sapiens").
            cell_type: Filter by cell type (e.g., "HeLa").
            repository: Filter by repository name (e.g., "OpenOrganelle").
            has_segmentation: If True, only return datasets with segmentations.

        Returns:
            List of matching DatasetEntry objects.
        """
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
