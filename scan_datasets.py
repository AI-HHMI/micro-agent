"""Scan the full OpenOrganelle S3 bucket to catalog all datasets with segmentations."""
import s3fs
import json

fs = s3fs.S3FileSystem(anon=True)

# List all datasets
all_datasets = fs.ls("janelia-cosem-datasets/")
dataset_dirs = [d.split("/")[-1] for d in all_datasets if not d.endswith(".md")]

results = []
for ds in sorted(dataset_dirs):
    n5_base = f"janelia-cosem-datasets/{ds}/{ds}.n5"
    try:
        # Check for EM data
        em_items = fs.ls(n5_base + "/em/")
        em_names = [i.split("/")[-1] for i in em_items if not i.endswith(".json")]
        if not em_names:
            continue

        # Check for labels/segmentations
        try:
            label_items = fs.ls(n5_base + "/labels/")
            seg_names = sorted([i.split("/")[-1] for i in label_items if i.split("/")[-1].endswith("_seg")])
        except FileNotFoundError:
            seg_names = []

        # Check scale levels for EM
        em_name = em_names[0]
        try:
            scales = fs.ls(n5_base + "/em/" + em_name + "/")
            scale_levels = sorted([s.split("/")[-1] for s in scales if s.split("/")[-1].startswith("s")])
        except:
            scale_levels = []

        results.append({
            "id": ds,
            "em_name": em_name,
            "scales": scale_levels,
            "seg_count": len(seg_names),
            "segs": seg_names,
        })
        status = "OK" if seg_names else "NO_SEG"
        print(f"{status:6s} {ds:40s} em={em_name:20s} scales={len(scale_levels)} segs={len(seg_names)}")
        if seg_names:
            # Print first few segs
            for s in seg_names[:5]:
                print(f"       - {s}")
            if len(seg_names) > 5:
                print(f"       ... and {len(seg_names) - 5} more")
    except FileNotFoundError:
        print(f"SKIP   {ds:40s} (no .n5 structure)")
    except Exception as e:
        print(f"ERROR  {ds:40s} {e}")

print(f"\n\nTotal datasets: {len(results)}")
print(f"With segmentations: {len([r for r in results if r['seg_count'] > 0])}")

# Save full results as JSON
with open("dataset_scan.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved to dataset_scan.json")
