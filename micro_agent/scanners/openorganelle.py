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
            n5_base = f"{bucket}/{ds_id}/{ds_id}.n5"
            try:
                em_items = self._fs.ls(n5_base + "/em/")
                em_names = [i.split("/")[-1] for i in em_items if not i.endswith(".json")]
                if not em_names:
                    continue
                em_name = em_names[0]
            except (FileNotFoundError, Exception):
                continue

            organelles: list[str] = []
            try:
                label_items = self._fs.ls(n5_base + "/labels/")
                seg_names = [
                    i.split("/")[-1] for i in label_items
                    if i.split("/")[-1].endswith("_seg")
                ]
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
                modality_class="em",
            ))

        return results

    async def validate_access(self, dataset: DiscoveredDataset) -> str:
        try:
            path = dataset.access_url.replace("s3://", "") + dataset.raw_path
            self._fs.info(path)
            return "verified"
        except Exception:
            return "failed"
