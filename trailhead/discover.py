"""Auto-discovery agent for public microscopy datasets.

Scans multiple repositories and the web to find datasets with raw EM data
and (ideally) paired segmentations. Outputs discovered_datasets.json entries
ready for the Registry to load.

Usage:
    python -m trailhead.discover [--output discovered_datasets.json]

Strategies:
1. API scanning — structured queries against known repos (OpenOrganelle, EMPIAR, IDR)
2. Web & literature search — find papers/preprints with public datasets
3. Cross-reference mining — follow links between repos
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import s3fs


@dataclass
class DiscoveredDataset:
    """A dataset found by the discovery agent."""

    id: str
    repository: str
    title: str
    organism: str = ""
    cell_type: str = ""
    imaging_modality: str = ""
    organelles: list[str] = field(default_factory=list)
    has_segmentation: bool = False
    has_raw: bool = True
    data_format: str = ""
    access_url: str = ""
    raw_path: str = ""
    segmentation_paths: dict[str, str] = field(default_factory=dict)
    provenance: str = ""  # How/where it was found
    supports_random_access: bool = True  # False for catalog-only entries without data paths
    # Multi-channel / fluorescence fields
    num_channels: int = 1
    channel_names: list[str] = field(default_factory=list)
    wavelengths_nm: list[float] = field(default_factory=list)
    fluorophores: list[str] = field(default_factory=list)
    bit_depth: int = 8
    modality_class: str = ""  # "em" | "fluorescence" | "correlative"
    validation_status: str = "pending"  # "verified" | "failed" | "pending"


# ---------------------------------------------------------------------------
# Strategy 1: OpenOrganelle S3 scan
# ---------------------------------------------------------------------------

def scan_openorganelle(fs: s3fs.S3FileSystem | None = None) -> list[DiscoveredDataset]:
    """Scan the OpenOrganelle S3 bucket for all datasets."""
    if fs is None:
        fs = s3fs.S3FileSystem(anon=True)

    bucket = "janelia-cosem-datasets"
    results: list[DiscoveredDataset] = []

    try:
        all_dirs = fs.ls(bucket)
    except Exception as e:
        print(f"  [OpenOrganelle] Failed to list bucket: {e}")
        return results

    dataset_ids = [d.split("/")[-1] for d in all_dirs if not d.endswith(".md")]

    for ds_id in sorted(dataset_ids):
        n5_base = f"{bucket}/{ds_id}/{ds_id}.n5"
        try:
            em_items = fs.ls(n5_base + "/em/")
            em_names = [i.split("/")[-1] for i in em_items if not i.endswith(".json")]
            if not em_names:
                continue
            em_name = em_names[0]
        except (FileNotFoundError, Exception):
            continue

        # Check for segmentations
        organelles: list[str] = []
        try:
            label_items = fs.ls(n5_base + "/labels/")
            seg_names = [i.split("/")[-1] for i in label_items if i.split("/")[-1].endswith("_seg")]
            organelles = [s.replace("_seg", "") for s in sorted(seg_names)]
        except FileNotFoundError:
            pass

        n5_rel = f"{ds_id}/{ds_id}.n5"
        results.append(DiscoveredDataset(
            id=ds_id,
            repository="OpenOrganelle",
            title=ds_id,
            data_format="n5",
            imaging_modality="FIB-SEM" if "fibsem" in em_name else "TEM",
            access_url=f"s3://{bucket}/{ds_id}/",
            raw_path=f"{n5_rel}/em/{em_name}",
            organelles=organelles,
            has_segmentation=len(organelles) > 0,
            segmentation_paths={o: f"{n5_rel}/labels/{o}_seg" for o in organelles},
            provenance="S3 bucket scan of janelia-cosem-datasets",
        ))

    return results


# ---------------------------------------------------------------------------
# Strategy 2: EMPIAR API scan
# ---------------------------------------------------------------------------

EMPIAR_API = "https://www.ebi.ac.uk/empiar/api"

def scan_empiar(max_entries: int = 50) -> list[DiscoveredDataset]:
    """Scan recent EMPIAR entries for EM datasets."""
    results: list[DiscoveredDataset] = []

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{EMPIAR_API}/entries/")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"  [EMPIAR] API error: {e}")
        return results

    entries = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = list(entries.values())[:max_entries]

    for entry_data in entries[:max_entries]:
        if isinstance(entry_data, dict):
            entry_id = str(entry_data.get("name", entry_data.get("id", "")))
            title = entry_data.get("title", "")
            organism = entry_data.get("organism", "")

            # Check if it looks like EM data
            modality = entry_data.get("imaging_modality", "")

            results.append(DiscoveredDataset(
                id=entry_id,
                repository="EMPIAR",
                title=title[:120] if title else entry_id,
                organism=organism,
                imaging_modality=modality,
                has_segmentation=False,
                data_format="mrc",
                access_url=f"https://www.ebi.ac.uk/empiar/entry/{entry_id}/",
                provenance="EMPIAR REST API scan",
            ))

    return results


# ---------------------------------------------------------------------------
# Strategy 3: IDR OME-Zarr scan
# ---------------------------------------------------------------------------

IDR_S3_ENDPOINT = "https://uk1s3.embassy.ebi.ac.uk"

def scan_idr() -> list[DiscoveredDataset]:
    """Check for known IDR datasets with OME-Zarr access."""
    results: list[DiscoveredDataset] = []

    # IDR doesn't have a simple "list all" API for OME-Zarr.
    # We check known OME-Zarr endpoints and query the MAPR API.
    try:
        with httpx.Client(timeout=30.0) as client:
            # Check IDR search for volume EM datasets
            resp = client.get(
                "https://idr.openmicroscopy.org/api/v0/m/screens/",
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                screens = resp.json().get("data", [])
                for screen in screens[:20]:
                    results.append(DiscoveredDataset(
                        id=str(screen.get("id", "")),
                        repository="IDR",
                        title=screen.get("name", ""),
                        data_format="ome-zarr",
                        access_url=f"https://idr.openmicroscopy.org/webclient/?show=screen-{screen.get('id', '')}",
                        provenance="IDR screens API",
                    ))
    except Exception as e:
        print(f"  [IDR] API error: {e}")

    return results


# ---------------------------------------------------------------------------
# Strategy 4: BioImage Archive scan
# ---------------------------------------------------------------------------

def scan_bioimage_archive() -> list[DiscoveredDataset]:
    """Check BioImage Archive for EM datasets with OME-Zarr."""
    results: list[DiscoveredDataset] = []

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                "https://www.ebi.ac.uk/biostudies/api/v1/search",
                params={
                    "query": "electron microscopy segmentation",
                    "pageSize": 20,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", [])
                for hit in hits:
                    results.append(DiscoveredDataset(
                        id=hit.get("accession", ""),
                        repository="BioImage Archive",
                        title=hit.get("title", ""),
                        access_url=f"https://www.ebi.ac.uk/biostudies/studies/{hit.get('accession', '')}",
                        provenance="BioStudies API search for 'electron microscopy segmentation'",
                    ))
    except Exception as e:
        print(f"  [BioImage Archive] API error: {e}")

    return results


# ---------------------------------------------------------------------------
# Main discovery orchestrator
# ---------------------------------------------------------------------------

def run_discovery(
    output_path: str = "discovered_datasets.json",
    use_new_scanners: bool = True,
    validate: bool = False,
) -> list[DiscoveredDataset]:
    """Run all discovery strategies and save results.

    Args:
        output_path: Path to write discovered datasets JSON.
        use_new_scanners: If True, use the async scanner module (8 sources
            including fluorescence). If False, use the legacy 4-source scan.
        validate: If True, validate accessibility of each discovered dataset.
    """
    if use_new_scanners:
        from trailhead.scanners import run_all_scanners
        print("Discovery agent starting (all sources)...")
        print()
        all_discovered = run_all_scanners(validate=validate)
    else:
        # Legacy mode: original 4 sources
        all_discovered = []
        print("Discovery agent starting (legacy mode)...")
        print()

        print("[1/4] Scanning OpenOrganelle S3 bucket...")
        oo_results = scan_openorganelle()
        print(f"  Found {len(oo_results)} datasets ({sum(1 for r in oo_results if r.has_segmentation)} with segmentations)")
        all_discovered.extend(oo_results)

        print("[2/4] Scanning EMPIAR API...")
        empiar_results = scan_empiar()
        print(f"  Found {len(empiar_results)} entries")
        all_discovered.extend(empiar_results)

        print("[3/4] Scanning IDR...")
        idr_results = scan_idr()
        print(f"  Found {len(idr_results)} entries")
        all_discovered.extend(idr_results)

        print("[4/4] Scanning BioImage Archive...")
        bia_results = scan_bioimage_archive()
        print(f"  Found {len(bia_results)} entries")
        all_discovered.extend(bia_results)

    # Save results
    output = [asdict(d) for d in all_discovered]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    em_count = sum(1 for d in all_discovered if d.modality_class == "em")
    fluor_count = sum(1 for d in all_discovered if d.modality_class == "fluorescence")
    seg_count = sum(1 for d in all_discovered if d.has_segmentation)
    verified = sum(1 for d in all_discovered if d.validation_status == "verified")

    print(f"\nTotal discovered: {len(all_discovered)} datasets")
    print(f"  EM: {em_count}  |  Fluorescence: {fluor_count}  |  Other: {len(all_discovered) - em_count - fluor_count}")
    print(f"  With segmentations: {seg_count}")
    if validate:
        print(f"  Verified accessible: {verified}")
    print(f"\nResults saved to {output_path}")

    return all_discovered


# ---------------------------------------------------------------------------
# CLI entry point: pixi run discover
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = "discovered_datasets.json"
    use_legacy = False
    do_validate = False

    args = sys.argv[1:]
    while args:
        if args[0] == "--output" and len(args) > 1:
            output = args[1]
            args = args[2:]
        elif args[0] == "--legacy":
            use_legacy = True
            args = args[1:]
        elif args[0] == "--validate":
            do_validate = True
            args = args[1:]
        else:
            output = args[0]
            args = args[1:]

    run_discovery(output, use_new_scanners=not use_legacy, validate=do_validate)
