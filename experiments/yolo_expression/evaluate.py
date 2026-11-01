# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate the selected YOLO checkpoint and baseline on the same fixed test subset."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from expression_lab.backends import InferenceBackend, ModelSpec
from expression_lab.evaluation import metrics, read_manifest
from expression_lab.labels import RAW_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evaluation report")
    reference = json.loads(args.reference_report.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if digest != reference["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    counts = {
        split: sum(r["split"] == split for r in rows) for split in ["train", "calibration", "test"]
    }
    if counts != reference["split_counts"]:
        raise ValueError("Split membership changed")
    rows = [r for r in rows if r["split"] == "test"]
    truth = np.array([RAW_LABELS.index(r["label"]) for r in rows])
    report = {
        "manifest_sha256": digest,
        "split_counts": counts,
        "preprocessing": "Whole face RGB224, no alignment or TTA, model-specific normalization",
        "threshold": 0.5,
        "margin": 0.08,
        "cpu_benchmark": "Single-face predict, OpenCV preprocessing + ONNX Runtime, two threads, excludes decoding/detection",
        "limitations": [
            "Previously reported test subset, not a fresh external benchmark",
            "Subject separation and pretraining overlap unknown",
            "No demographic fairness annotations or webcam accuracy measurement",
            "Visible expression estimates do not establish internal feelings",
        ],
    }
    for name, manifest in [("baseline", args.baseline), ("candidate", args.candidate)]:
        spec = ModelSpec.load(manifest)
        backend = InferenceBackend(spec)
        crop = cv2.imread(str(rows[0]["path"]))
        for _ in range(20):
            backend.predict(crop)
        probabilities, durations = [], []
        for i, row in enumerate(rows):
            crop = cv2.imread(str(row["path"]))
            if crop is None:
                raise ValueError(f"Unreadable image: {row['path']}")
            start = time.perf_counter()
            probabilities.append(backend.predict(crop))
            durations.append(time.perf_counter() - start)
            if (i + 1) % 500 == 0:
                print(f"{name}: {i + 1}/{len(rows)}", flush=True)
        report[name] = {
            "sha256": spec.sha256,
            **metrics(truth, probabilities, threshold=0.5, margin=0.08),
            "cpu_mean_ms": float(np.mean(durations) * 1000),
            "cpu_p95_ms": float(np.percentile(durations, 95) * 1000),
            "cpu_faces_per_second": float(1 / np.mean(durations)),
        }
        print(json.dumps({name: report[name]}), flush=True)
        del backend
    report["production_model_changed"] = False
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
