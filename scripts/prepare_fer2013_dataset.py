"""Prepare bounded extra training data, screening known held-out perceptual matches."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def image_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8))
    return np.uint64(int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("per-class must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.source / "source.json").read_text())
    reference = json.loads(args.reference_report.read_text())
    digest = hashlib.sha256(args.heldout_manifest.read_bytes()).hexdigest()
    if digest != reference["dataset_manifest_sha256"]:
        raise ValueError("Original dataset manifest changed")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    heldout = []
    counts = Counter()
    with args.heldout_manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = (args.heldout_manifest.parent / row["path"]).resolve()
            if path in excluded or row["split"] not in ("calibration", "test"):
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Cannot read existing held-out image")
            heldout.append(image_hash(image))
            counts[row["split"]] += 1
    if any(counts[s] != reference["split_counts"][s] for s in ("calibration", "test")):
        raise ValueError("Original held-out split counts changed")
    import pandas as pd
    from fastparquet import ParquetFile

    frames = {}
    for split in ("train", "test"):
        path = args.source / f"{split}.parquet"
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != source["files"][split]["sha256"]:
            raise ValueError("Source Parquet checksum mismatch")
        metadata = json.loads(ParquetFile(path).key_value_metadata["huggingface"])
        if metadata["info"]["features"]["labels"]["names"] != source["labels"]:
            raise ValueError("Pinned Parquet label order mismatch")
        frame = pd.read_parquet(path, engine="fastparquet")
        if len(frame) != source["rows"][split] or set(frame.columns) != {"img_bytes", "labels"}:
            raise ValueError("Unexpected source schema or row count")
        if not set(frame["labels"].unique()) <= set(range(len(source["labels"]))):
            raise ValueError("Unexpected source labels")
        frames[split] = frame
    manifests = {"train": [], "test": []}
    for index, row in enumerate(frames["test"].itertuples(index=False)):
        data = row.img_bytes
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode source test image")
        heldout.append(image_hash(image))
        label = source["labels"][int(row.labels)]
        path = args.output / "images/test" / f"{index:05d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifests["test"].append((str(path.relative_to(args.output)), label, "test"))
    bank = np.asarray(heldout, dtype=np.uint64)
    rng = np.random.default_rng(42)
    chosen, skipped = Counter(), Counter()
    content_hashes = set()
    for index in rng.permutation(len(frames["train"])):
        row = frames["train"].iloc[int(index)]
        label = source["labels"][int(row["labels"])]
        if chosen[label] >= args.per_class:
            continue
        data = row["img_bytes"]
        digest_bytes = hashlib.sha256(data).hexdigest()
        if digest_bytes in content_hashes:
            skipped["duplicate_training_bytes"] += 1
            continue
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            skipped["undecodable"] += 1
            continue
        value = image_hash(image)
        if np.any(np.bitwise_count(bank ^ value) <= 4):
            skipped["heldout_perceptual_match"] += 1
            continue
        content_hashes.add(digest_bytes)
        path = args.output / "images/train" / f"{int(index):05d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifests["train"].append((str(path.relative_to(args.output)), label, "train"))
        chosen[label] += 1
    for split, rows in manifests.items():
        with (args.output / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "split"])
            writer.writerows(rows)
    report = {
        "source": source,
        "original_heldout_manifest_sha256": digest,
        "original_heldout_counts": dict(counts),
        "maximum_extra_training_per_class": args.per_class,
        "seed": 42,
        "extra_training_counts": dict(chosen),
        "test_images": len(manifests["test"]),
        "skipped_training": dict(skipped),
        "limitations": [
            "Perceptual hashes do not establish person identity separation",
            "Pretraining overlap and original training overlap with external test are unknown",
            "External test images are not added to training",
        ],
    }
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
