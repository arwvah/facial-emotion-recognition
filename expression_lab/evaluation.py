"""Dataset evaluation, model/FPS comparison, and held-out temperature fitting.

CSV columns: path,label,split,group,subject_id. Group/subject are optional supplied annotations;
the application never estimates demographics. FER+/AffectNet crops can be referenced directly.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from .backends import scale_probabilities
from .emotion import EmotionRecognizer
from .labels import RAW_LABELS


def read_manifest(path: Path, require_labels=True) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not ({"path", "label"} if require_labels else {"path"}) <= set(reader.fieldnames or ()):
            raise ValueError("Dataset CSV requires path and label columns")
        rows = []
        seen = {}
        seen_hashes: dict[str, str] = {}
        subjects: dict[str, str] = {}
        aliases = {"happiness": "happy", "sadness": "sad", "anger": "angry"}
        for raw_row in reader:
            row: dict[str, Any] = dict(raw_row)
            row["path"] = (path.parent / row["path"]).resolve()
            row["label"] = aliases.get(row.get("label", ""), row.get("label", ""))
            row["split"] = row.get("split") or "test"
            if (require_labels and row["label"] not in RAW_LABELS) or row["split"] not in (
                "test",
                "calibration",
                "train",
            ):
                raise ValueError("Unknown label or split in dataset")
            if not row["path"].is_file():
                raise FileNotFoundError(row["path"])
            if row["path"] in seen:
                raise ValueError(
                    "Duplicate image path; do not reuse calibration/train/test samples"
                )
            seen[row["path"]] = row["split"]
            subject = row.get("subject_id")
            if subject:
                if subject in subjects and subjects[subject] != row["split"]:
                    raise ValueError(
                        "Subject appears in different splits; use subject-disjoint data"
                    )
                subjects[subject] = row["split"]
            digest = hashlib.sha256(row["path"].read_bytes()).hexdigest()
            if digest in seen_hashes and seen_hashes[digest] != row["split"]:
                raise ValueError("Identical image contents appear in different splits")
            seen_hashes[digest] = row["split"]
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def metrics(labels, probabilities, threshold=0.5, rejected=None, margin=0.0) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if (
        probabilities.shape != (len(labels), 8)
        or not len(labels)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Metrics require N labels and finite Nx8 probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probabilities")
    if ((labels < 0) | (labels > 7)).any():
        raise ValueError("Invalid class index")
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    if not 0 <= margin <= 1:
        raise ValueError("Expression margin must be in [0,1]")
    accepted = (confidence >= threshold) & (
        confidence - np.partition(probabilities, -2, axis=1)[:, -2] >= margin
    )
    if rejected is not None:
        accepted &= ~np.asarray(rejected, dtype=bool)
    output = np.where(accepted, prediction, 8)
    confusion = np.zeros((8, 9), dtype=int)
    np.add.at(confusion, (labels, output), 1)
    per_class = {}
    for index, name in enumerate(RAW_LABELS):
        tp = int(confusion[index, index])
        support = int((labels == index).sum())
        precision = tp / max(1, int((output == index).sum()))
        recall = tp / max(1, support)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "support": support,
        }
    present = [value for value in per_class.values() if value["support"]]
    ece = 0.0
    for index in range(15):
        mask = (confidence >= index / 15) & (
            confidence < (index + 1) / 15 if index < 14 else confidence <= 1
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float((prediction[mask] == labels[mask]).mean()) - float(confidence[mask].mean())
            )
    truth = np.eye(8)[labels]
    return {
        "samples": len(labels),
        "accuracy": float((output == labels).mean()),
        "raw_top1_accuracy": float((prediction == labels).mean()),
        "coverage": float(accepted.mean()),
        "selective_accuracy": float((prediction[accepted] == labels[accepted]).mean())
        if accepted.any()
        else None,
        "macro_f1_present_classes": float(np.mean([x["f1"] for x in present])),
        "balanced_accuracy_present_classes": float(np.mean([x["recall"] for x in present])),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        ),
        "brier": float(((probabilities - truth) ** 2).sum(axis=1).mean()),
        "ece_15_bins": ece,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "rows": list(RAW_LABELS),
        "columns": [*RAW_LABELS, "uncertain"],
    }


def fairness_metrics(rows, labels, probabilities, threshold, rejected, minimum=20, margin=0.0):
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        mask = np.array([row.get("group") == group for row in rows])
        count = int(mask.sum())
        result[group] = (
            {"samples": count, "status": "insufficient_samples"}
            if count < minimum
            else {
                "status": "evaluated",
                **metrics(
                    np.asarray(labels)[mask],
                    np.asarray(probabilities)[mask],
                    threshold,
                    np.asarray(rejected)[mask],
                    margin,
                ),
            }
        )
    evaluated = [value["accuracy"] for value in result.values() if value["status"] == "evaluated"]
    return {
        "groups": result,
        "minimum_samples": minimum,
        "accuracy_gap": max(evaluated) - min(evaluated) if len(evaluated) > 1 else None,
        "status": "not_assessed_no_group_annotations" if not groups else "group_metrics_only",
        "limitations": "User-supplied group labels only; unequal samples, class balance and confidence intervals must be considered. This is not a fairness certification.",
    }


def fit_temperature(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if probabilities.shape != (len(labels), 8) or not len(labels):
        raise ValueError("Calibration requires labeled Nx8 predictions")

    def loss(log_temperature):
        scaled = np.array(
            [scale_probabilities(row, math.exp(log_temperature)) for row in probabilities]
        )
        return float(-np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    # Bounded one-dimensional search; the baseline is always an admissible fallback.
    left, right = math.log(0.05), math.log(20)
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(70):
        a = right - ratio * (right - left)
        b = left + ratio * (right - left)
        if loss(a) < loss(b):
            right = b
        else:
            left = a
    candidate = (left + right) / 2
    if loss(candidate) >= loss(0):
        candidate = 0
    return math.exp(candidate), loss(0), loss(candidate)


def collect(rows, recognizer, input_mode="crop", quality_checks=False):
    predictions = []
    rejected = []
    latencies = []
    for row in rows:
        start = time.perf_counter()
        image = cv2.imread(str(row["path"]))
        if image is None:
            raise ValueError(f"Cannot decode dataset image: {row['path']}")
        if input_mode == "frame":
            faces = recognizer.detect(image)
            if not faces:
                scores = np.full(8, 0.125)
                bad = True
            else:
                scores = np.array([faces[0].probabilities[name] for name in RAW_LABELS])
                bad = faces[0].uncertainty_reason == "image_quality"
        else:
            scores = recognizer.predict_crop(image)
            bad = quality_checks and not recognizer.quality_policy.measure(image)["accepted"]
        latencies.append(time.perf_counter() - start)
        predictions.append(scores)
        rejected.append(bad)
    return np.array(predictions), np.array(rejected), latencies


def benchmark(
    rows, configuration, threshold=0.5, input_mode="crop", min_group=20, quality_checks=False
):
    allowed = {
        "engine",
        "provider",
        "model_spec",
        "expression_model",
        "expression_margin",
        "align",
        "tta",
        "temperature",
        "calibration",
    }
    if set(configuration) - allowed:
        raise ValueError("Unknown benchmark model setting")
    model_options = dict(configuration)
    calibration = model_options.pop("calibration", None)
    recognizer = EmotionRecognizer(
        threshold, "haar" if input_mode == "crop" else "yunet", **model_options
    )
    if calibration:
        from .configuration import apply_calibration

        apply_calibration(
            recognizer,
            SimpleNamespace(
                calibration=Path(calibration), align=recognizer.align, tta=recognizer.tta
            ),
        )
    if input_mode == "crop" and recognizer.align:
        raise ValueError(
            "Crop evaluation cannot realign without landmarks; set align=false or use input-mode frame"
        )
    # Exclude first-run initialization from the timed samples.
    collect(rows[:1], recognizer, input_mode, quality_checks)
    probabilities, rejected, latencies = collect(rows, recognizer, input_mode, quality_checks)
    labels = np.array([RAW_LABELS.index(row["label"]) for row in rows])
    result = metrics(labels, probabilities, threshold, rejected, recognizer.expression_margin)
    result.update(
        {
            "engine": configuration.get("engine", "onnx"),
            "model_sha256": recognizer.backend.spec.sha256,
            "temperature": recognizer.temperature,
            "execution_providers": recognizer.backend.session.get_providers()
            if recognizer.backend.engine == "onnx"
            else None,
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in configuration.items()
            },
            "input_mode": input_mode,
            "fps_including_decode": len(rows) / max(sum(latencies), 1e-12),
            "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
            "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
            "fairness": fairness_metrics(
                rows,
                labels,
                probabilities,
                threshold,
                rejected,
                min_group,
                recognizer.expression_margin,
            ),
        }
    )
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "compare", "calibrate"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", type=Path, help="JSON list of named model configurations")
    parser.add_argument("--model-spec", type=Path)
    parser.add_argument("--expression-model", choices=("ferplus", "hsemotion"), default="ferplus")
    parser.add_argument("--expression-margin", type=float, default=0.0)
    parser.add_argument(
        "--calibration", type=Path, help="Evaluate with an already fitted temperature artifact"
    )
    parser.add_argument("--engine", choices=("onnx", "opencv", "openvino"), default="onnx")
    parser.add_argument("--input-mode", choices=("crop", "frame"), default="crop")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quality-checks", action="store_true")
    parser.add_argument("--min-group", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new path")
        if not 0 <= args.threshold <= 1 or args.min_group < 1:
            raise ValueError("Invalid threshold or group size")
        rows = read_manifest(args.manifest)
        configuration = {
            "engine": args.engine,
            "align": args.align,
            "tta": args.tta,
            "expression_model": args.expression_model,
            "expression_margin": args.expression_margin,
        }
        if args.model_spec and args.expression_model != "ferplus":
            raise ValueError("Choose --expression-model or --model-spec")
        if args.model_spec:
            configuration["model_spec"] = args.model_spec
        if args.calibration:
            if args.command == "calibrate":
                raise ValueError("Cannot fit temperature on already calibrated scores")
            configuration["calibration"] = args.calibration
        data_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if args.command == "calibrate":
            selected = [row for row in rows if row["split"] == "calibration"]
            if not selected:
                raise ValueError("Calibration requires rows explicitly marked split=calibration")
            if args.input_mode == "crop" and args.align:
                raise ValueError("Crop calibration requires --no-align")
            recognizer = EmotionRecognizer(
                args.threshold, "haar" if args.input_mode == "crop" else "yunet", **configuration
            )
            probabilities, rejected, _ = collect(
                selected, recognizer, args.input_mode, args.quality_checks
            )
            accepted = ~rejected
            labels = np.array([RAW_LABELS.index(row["label"]) for row in selected])
            temperature, before, after = fit_temperature(probabilities[accepted], labels[accepted])
            write_json(
                args.output,
                {
                    "schema_version": 1,
                    "temperature": temperature,
                    "nll_before": before,
                    "nll_after": after,
                    "samples": int(accepted.sum()),
                    "model_sha256": recognizer.backend.spec.sha256,
                    "model_contract": recognizer.backend.spec.signature(),
                    "preprocessing": {"align": args.align, "tta": args.tta},
                    "dataset_manifest_sha256": data_hash,
                    "note": "Fit on calibration split only. Evaluate independently on the untouched test split.",
                },
            )
        else:
            selected = [row for row in rows if row["split"] == "test"]
            if not selected:
                raise ValueError("Evaluation requires test-split samples")
            if args.command == "compare":
                if not args.models:
                    raise ValueError("compare requires --models JSON")
                configurations = json.loads(args.models.read_text())
            else:
                configurations = [{"name": "selected", **configuration}]
            results = []
            for item in configurations:
                item = dict(item)
                name = item.pop("name")
                item.setdefault("align", args.align)
                item.setdefault("tta", args.tta)
                if item.get("model_spec"):
                    item["model_spec"] = (
                        (args.models.parent / item["model_spec"]).resolve()
                        if args.models
                        else Path(item["model_spec"])
                    )
                if item.get("calibration"):
                    item["calibration"] = (
                        (args.models.parent / item["calibration"]).resolve()
                        if args.models
                        else Path(item["calibration"])
                    )
                results.append(
                    {
                        "name": name,
                        **benchmark(
                            selected,
                            item,
                            args.threshold,
                            args.input_mode,
                            args.min_group,
                            args.quality_checks,
                        ),
                    }
                )
            write_json(
                args.output,
                {"dataset_manifest_sha256": data_hash, "split": "test", "results": results},
            )
        print(f"Wrote {args.output}")
        return 0
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
