# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a YOLO classifier dataset with the existing exclusions and held-out test isolation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from expression_lab.evaluation import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    reference = json.loads(args.reference_report.read_text())
    if (
        hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        != reference["dataset_manifest_sha256"]
    ):
        raise ValueError("Dataset manifest changed")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    counts = {
        split: sum(r["split"] == split for r in rows) for split in ["train", "calibration", "test"]
    }
    if counts != reference["split_counts"]:
        raise ValueError("Reference split membership changed")
    args.output.mkdir(parents=True)
    for row in rows:
        if row["split"] == "test":
            continue
        split = "val" if row["split"] == "calibration" else "train"
        folder = args.output / split / row["label"]
        folder.mkdir(parents=True, exist_ok=True)
        source = row["path"]
        target = folder / source.name
        if target.exists():
            raise ValueError("Duplicate destination filename")
        os.link(source, target)
    (args.output / "preparation.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "test_images_not_linked_into_training_dataset": True,
                "source_manifest_sha256": reference["dataset_manifest_sha256"],
                "excluded_count": len(excluded),
            },
            indent=2,
        )
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
