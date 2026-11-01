"""Expand from the pinned source training archive, preserving fixed held-out membership."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

LABELS = ("angry", "surprise", "contempt", "happy", "neutral", "fear", "sad", "disgust")
SOURCE_SHA256 = "3eadf11aa0c7bac6edfa6a11d829632ad6e62e51644b7e7ced31d2a0e6ec767f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--heldout-manifest", type=Path, default=Path("datasets/expressions/manifest.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source training archive checksum mismatch")
    args.output.mkdir(parents=True)
    rows = []
    seen = set()
    counts = dict.fromkeys(LABELS, 0)
    duplicates = 0
    with args.heldout_manifest.open() as stream:
        for old in csv.DictReader(stream):
            if old["split"] == "train":
                continue
            path = (args.heldout_manifest.parent / old["path"]).resolve()
            seen.add(hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(
                {
                    "path": str(path),
                    "label": old["label"],
                    "split": old["split"],
                    "subject_id": "",
                    "group": "",
                }
            )
    df = pd.read_parquet(args.source)
    for idx, row in df.iterrows():
        data = row["image.bytes"]
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        label = LABELS[int(row["label"])]
        folder = args.output / "train" / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{idx:06d}.png"
        path.write_bytes(data)
        counts[label] += 1
        rows.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "split": "train",
                "subject_id": "",
                "group": "",
            }
        )
    with (args.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["path", "label", "split", "subject_id", "group"]
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source_sha256": digest,
        "source_rows": len(df),
        "training_counts": counts,
        "exact_duplicates_excluded": duplicates,
        "heldout_policy": "Prior validation/test membership preserved. Trainer applies perceptual similarity exclusions. Person identities are not asserted.",
    }
    (args.output / "preparation-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
