"""Scanner for OpenOrganelle S3 bucket."""

from __future__ import annotations

import s3fs

from micro_agent.discover import DiscoveredDataset
from micro_agent.scanners.base import BaseScanner


class OpenOrganelleScanner(BaseScanner):
    """Scan the OpenOrganelle S3 bucket for FIB-SEM datasets."""

    name = "OpenOrganelle"

    def __init__(self) -> None:
        self._fs = s3fs.S3FileSystem(anon=True)

    async def scan(self, limit: int = 50) -> list[DiscoveredDataset]:
        bucket = "janelia-cosem-datasets"
        results: list[DiscoveredDataset] = []

        try:
            all_dirs = self._fs.ls(bucket)
        except Exception as e:
            print(f"  [{self.name}] Failed to list bucket: {e}")
            return results

        dataset_ids = [d.split("/")[-1] for d in all_dirs if not d.endswith(".md")]

        for ds_id in sorted(dataset_ids)[:limit]:
            # Try zarr first, fall back to N5
            raw_path, em_name, data_format = self._find_raw(bucket, ds_id)
            if raw_path is None:
                continue

            organelles, seg_paths = self._find_segmentations(bucket, ds_id, data_format)

            results.append(DiscoveredDataset(
                id=ds_id,
                repository="OpenOrganelle",
                title=ds_id,
                data_format=data_format,
                imaging_modality="FIB-SEM" if "fibsem" in em_name else "TEM",
                access_url=f"s3://{bucket}/{ds_id}/",
                raw_path=raw_path,
                organelles=organelles,
                has_segmentation=len(organelles) > 0,
                segmentation_paths=seg_paths,
                provenance="S3 bucket scan of janelia-cosem-datasets",
                modality_class="em",
            ))

        return results

    def _find_raw(self, bucket: str, ds_id: str) -> tuple[str | None, str, str]:
        """Find raw EM path, preferring zarr over N5.

        Returns (relative_path, em_name, format) or (None, "", "") if not found.
        """
        # Try zarr: {id}/{id}.zarr/recon-1/em/
        zarr_base = f"{bucket}/{ds_id}/{ds_id}.zarr/recon-1"
        try:
            em_items = self._fs.ls(zarr_base + "/em/")
            em_names = [i.split("/")[-1] for i in em_items
                        if not i.split("/")[-1].startswith(".") and not i.endswith(".json")]
            if em_names:
                em_name = em_names[0]
                return f"{ds_id}/{ds_id}.zarr/recon-1/em/{em_name}", em_name, "zarr"
        except (FileNotFoundError, Exception):
            pass

        # Fall back to N5: {id}/{id}.n5/em/
        n5_base = f"{bucket}/{ds_id}/{ds_id}.n5"
        try:
            em_items = self._fs.ls(n5_base + "/em/")
            em_names = [i.split("/")[-1] for i in em_items
                        if not i.split("/")[-1].startswith(".") and not i.endswith(".json")]
            if em_names:
                em_name = em_names[0]
                return f"{ds_id}/{ds_id}.n5/em/{em_name}", em_name, "n5"
        except (FileNotFoundError, Exception):
            pass

        return None, "", ""

    def _find_segmentations(
        self, bucket: str, ds_id: str, raw_format: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Find segmentation paths, preferring the same format as raw."""
        organelles: list[str] = []
        seg_paths: dict[str, str] = {}

        # Try same format as raw first
        if raw_format == "zarr":
            bases = [
                (f"{bucket}/{ds_id}/{ds_id}.zarr/recon-1/labels/", f"{ds_id}/{ds_id}.zarr/recon-1/labels"),
                (f"{bucket}/{ds_id}/{ds_id}.n5/labels/", f"{ds_id}/{ds_id}.n5/labels"),
            ]
        else:
            bases = [
                (f"{bucket}/{ds_id}/{ds_id}.n5/labels/", f"{ds_id}/{ds_id}.n5/labels"),
                (f"{bucket}/{ds_id}/{ds_id}.zarr/recon-1/labels/", f"{ds_id}/{ds_id}.zarr/recon-1/labels"),
            ]

        for abs_base, rel_base in bases:
            try:
                label_items = self._fs.ls(abs_base)
                seg_names = [
                    i.split("/")[-1] for i in label_items
                    if i.split("/")[-1].endswith("_seg")
                ]
                if seg_names:
                    organelles = [s.replace("_seg", "") for s in sorted(seg_names)]
                    seg_paths = {o: f"{rel_base}/{o}_seg" for o in organelles}
                    return organelles, seg_paths
            except (FileNotFoundError, Exception):
                pass

        return organelles, seg_paths

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            path = dataset.access_url.replace("s3://", "") + dataset.raw_path
            self._fs.info(path)
            return "verified"
        except Exception:
            return "failed"
