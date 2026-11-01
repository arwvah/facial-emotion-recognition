"""Build a bounded COCO training subset; preserve existing validation/test membership."""

import argparse
import collections
import concurrent.futures
import hashlib
import heapq
import json
import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import ijson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, default=Path("datasets/objects"))
    parser.add_argument("--images-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.images_per_class < 1:
        parser.error("images-per-class must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.annotations.open("rb") as f:
        images = {im["id"]: im for im in ijson.items(f, "images.item", use_float=True)}
    with args.annotations.open("rb") as f:
        categories = list(ijson.items(f, "categories.item", use_float=True))
    with args.annotations.open("rb") as f:
        licenses = list(ijson.items(f, "licenses.item", use_float=True))
    with args.annotations.open("rb") as f:
        info = next(ijson.items(f, "info", use_float=True))
    pools = {c["id"]: [] for c in categories}
    members = {c["id"]: set() for c in categories}
    capacity = max(100, args.images_per_class)
    with args.annotations.open("rb") as f:
        for ann in ijson.items(f, "annotations.item", use_float=True):
            if ann.get("iscrowd", 0):
                continue
            cls = ann["category_id"]
            image_id = ann["image_id"]
            if image_id in members[cls]:
                continue
            key = int.from_bytes(
                hashlib.sha256(f"42:{cls}:{image_id}".encode()).digest()[:8], "big"
            )
            pool = pools[cls]
            if len(pool) < capacity:
                heapq.heappush(pool, (-key, image_id))
                members[cls].add(image_id)
            elif key < -pool[0][0]:
                _, removed = heapq.heapreplace(pool, (-key, image_id))
                members[cls].remove(removed)
                members[cls].add(image_id)
    chosen = set()
    for cls, pool in pools.items():
        chosen.update(
            image_id for _, image_id in sorted(pool, reverse=True)[: args.images_per_class]
        )
    annotations = []
    with args.annotations.open("rb") as f:
        for ann in ijson.items(f, "annotations.item", use_float=True):
            if ann["image_id"] in chosen:
                annotations.append(ann)
    folder = args.output / "train/images"
    folder.mkdir(parents=True, exist_ok=True)

    def download(image_id):
        info = images[image_id]
        p = folder / info["file_name"]
        if not p.exists():
            url = "http://images.cocodataset.org/train2017/" + info["file_name"]
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=40) as response:
                        data = response.read(15 * 1024 * 1024)
                    temp = p.with_suffix(".part")
                    temp.write_bytes(data)
                    temp.replace(p)
                    break
                except Exception:
                    if attempt == 2:
                        raise
        im = cv2.imread(str(p))
        if im is None or im.shape[:2] != (info["height"], info["width"]):
            raise ValueError(f"Invalid image: {p}")
        return image_id

    print(f"Downloading/validating {len(chosen)} training images", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for count, _ in enumerate(pool.map(download, sorted(chosen)), 1):
            if count % 100 == 0:
                print(f"Training images {count}/{len(chosen)}", flush=True)

    def pixel_hash(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        return hashlib.sha256(image.tobytes() + str(image.shape).encode()).hexdigest()

    seen = {
        pixel_hash(p)
        for split in ["val", "test"]
        for p in (args.heldout_root / split / "images").glob("*")
    }
    excluded = []
    for image_id in sorted(chosen):
        digest = pixel_hash(folder / images[image_id]["file_name"])
        if digest in seen:
            excluded.append(image_id)
        seen.add(digest)
    chosen.difference_update(excluded)
    annotations = [a for a in annotations if a["image_id"] in chosen]
    out = args.output / "annotations"
    out.mkdir(exist_ok=True)
    (out / "instances_train.json").write_text(
        json.dumps(
            {
                "info": info,
                "licenses": licenses,
                "images": [images[i] for i in sorted(chosen)],
                "annotations": annotations,
                "categories": categories,
            }
        )
    )
    for split in ["val", "test"]:
        shutil.copyfile(
            args.heldout_root / f"annotations/instances_{split}.json",
            out / f"instances_{split}.json",
        )
        dst = args.output / split / "images"
        dst.mkdir(parents=True, exist_ok=True)
        for p in (args.heldout_root / split / "images").glob("*"):
            target = dst / p.name
            if not target.exists():
                os.link(p, target)
    counts = collections.Counter(a["category_id"] for a in annotations if not a.get("iscrowd", 0))
    if any(not counts[c["id"]] for c in categories):
        raise ValueError("Expanded training subset is missing a target category")
    report = {
        "train_images": len(chosen),
        "train_boxes": len(annotations),
        "excluded_exact_pixel_duplicates": excluded,
        "class_counts": {c["name"]: counts[c["id"]] for c in categories},
        "source_annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "selection": f"Deterministic SHA-256 priority per category; {args.images_per_class} training images/category, union may overlap across categories.",
        "source_url": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "validation_test": "Hard links to prior fixed splits; no changes to membership.",
    }
    (args.output / "preparation-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"images": len(chosen), "boxes": len(annotations), "classes": len(counts)}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
