"""Select three-seed residual candidates on validation and test only the selected model."""

import argparse
import dataclasses
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import numpy_helper

from expression_lab.backends import InferenceBackend, ModelSpec
from expression_lab.evaluation import metrics, read_manifest
from expression_lab.labels import RAW_LABELS


def softmax(logits):
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.run / "selection-report.json"
    if report_path.exists():
        raise ValueError("Refusing to overwrite an existing selection report")
    plan = json.loads((args.run / "experiment-plan.json").read_text())
    spec = ModelSpec.load(args.baseline)
    original = onnx.load(spec.path)
    head = original.graph.node[-1]
    if head.op_type != "Gemm":
        raise ValueError("Expected the existing linear CNN baseline")
    base = {i.name: numpy_helper.to_array(i) for i in original.graph.initializer}
    reference = json.loads(args.reference_report.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if digest != reference["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "calibration", "test")}
    if counts != reference["split_counts"]:
        raise ValueError("Split counts changed")
    feature_path = args.feature_run / "features.npy"
    with feature_path.open("rb") as handle:
        feature_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    x = np.load(feature_path, mmap_mode="r")
    if x.shape != (len(rows) + counts["train"], base[head.input[1]].shape[1]):
        raise ValueError("Feature cache shape mismatch")
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "calibration"])
    xv = np.asarray(x[val])
    yv = np.array([spec.labels.index(rows[i]["label"]) for i in val])
    base_val = xv @ base[head.input[1]].T + base[head.input[2]]
    parameters, paths = {}, {"baseline": spec.path}
    names = ("residual_w1", "residual_b1", "residual_w2", "residual_b2")
    models = []
    for seed in plan["seeds"]:
        folder = args.run / f"seed-{seed}"
        training = json.loads((folder / "training-report.json").read_text())
        if (
            not training["test_deferred"]
            or training["starting_model_sha256"] != spec.sha256
            or training["manifest_sha256"] != digest
            or training["feature_cache_sha256"] != feature_sha
            or training["split_counts"] != counts
        ):
            raise ValueError("Candidate provenance/test isolation mismatch")
        for key in ("hidden", "dropout", "learning_rate"):
            if training[key] != plan[key]:
                raise ValueError(f"Candidate parameter changed: {key}")
        candidate = ModelSpec.load(folder / "model.json")
        if any(candidate.signature()[k] != v for k, v in spec.signature().items() if k != "sha256"):
            raise ValueError("Candidate preprocessing/label order mismatch")
        model = onnx.load(candidate.path)
        weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        for name, value in base.items():
            np.testing.assert_array_equal(weights[name], value)
        variant = f"seed-{seed}"
        parameters[variant] = [weights[n] for n in names]
        paths[variant] = candidate.path
        models.append(model)
    n = len(plan["seeds"])
    parameters["ensemble"] = [
        np.concatenate([parameters[f"seed-{s}"][0] for s in plan["seeds"]], axis=1),
        np.concatenate([parameters[f"seed-{s}"][1] for s in plan["seeds"]]),
        np.concatenate([parameters[f"seed-{s}"][2] / n for s in plan["seeds"]], axis=0),
        np.mean([parameters[f"seed-{s}"][3] for s in plan["seeds"]], axis=0),
    ]
    ensemble = models[0]
    for item in ensemble.graph.initializer:
        if item.name in names:
            value = parameters["ensemble"][names.index(item.name)]
            item.CopyFrom(numpy_helper.from_array(value.astype(np.float32), item.name))
    onnx.checker.check_model(ensemble)
    paths["ensemble"] = args.run / "ensemble.onnx"
    onnx.save(ensemble, paths["ensemble"])
    manifest = dataclasses.asdict(spec)
    manifest.update(
        path="ensemble.onnx", sha256=hashlib.sha256(paths["ensemble"].read_bytes()).hexdigest()
    )
    (args.run / "ensemble.json").write_text(json.dumps(manifest, indent=2))

    def logits(variant, features, baseline):
        if variant == "baseline":
            return baseline
        a, b, c, d = parameters[variant]
        return baseline + np.maximum(features @ a + b, 0) @ c + d

    validation = {}
    for variant in paths:
        p = softmax(logits(variant, xv, base_val))
        validation[variant] = {
            "nll": float(-np.log(np.maximum(p[np.arange(len(val)), yv], 1e-12)).mean()),
            "top1": float((p.argmax(1) == yv).mean()),
        }
    selected = min(validation, key=lambda v: validation[v]["nll"])
    if validation[selected]["nll"] >= validation["baseline"]["nll"] - 0.0001:
        selected = "baseline"
    selected_path = args.run / "selected.onnx"
    shutil.copy2(paths[selected], selected_path)
    manifest.update(
        path=selected_path.name, sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest()
    )
    selected_manifest = args.run / "selected.json"
    selected_manifest.write_text(json.dumps(manifest, indent=2))
    backend = InferenceBackend(ModelSpec.load(selected_manifest))
    ensemble_backend = InferenceBackend(ModelSpec.load(args.run / "ensemble.json"))
    order = [spec.labels.index(label) for label in RAW_LABELS]
    for index in (0, len(val) // 2, len(val) - 1):
        crop = cv2.imread(str(rows[int(val[index])]["path"]))
        for variant, engine in [(selected, backend), ("ensemble", ensemble_backend)]:
            expected = softmax(logits(variant, xv[index : index + 1], base_val[index : index + 1]))[
                0
            ]
            np.testing.assert_allclose(engine.predict(crop), expected[order], atol=1e-5)
    gate = plan["promotion_gate"]
    gain = validation[selected]["top1"] - validation["baseline"]["top1"]
    report = {
        "selection": plan["selection"],
        "starting_model_sha256": spec.sha256,
        "manifest_sha256": digest,
        "feature_cache_sha256": feature_sha,
        "split_counts": counts,
        "validation": validation,
        "selected_variant": selected,
        "validation_top1_gain": gain,
        "promotion_gate": gate,
        "accepted": False,
        "production_model_changed": False,
        "test_status": "not_evaluated_validation_gate_failed",
        "limitations": [
            "Validation has been reused for earlier model experiments",
            "The fixed test subset was reported earlier; not an independent external benchmark",
            "Identity separation and pretraining overlap unknown",
            "Visible expression labels do not establish feelings",
        ],
    }
    if selected != "baseline" and gain >= gate["validation_top1_minimum_gain"]:
        test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
        xt = np.asarray(x[test])
        bt = xt @ base[head.input[1]].T + base[head.input[2]]
        truth = np.array([RAW_LABELS.index(rows[i]["label"]) for i in test])
        for variant, key in [("baseline", "baseline_test"), (selected, "candidate_test")]:
            report[key] = metrics(
                truth, softmax(logits(variant, xt, bt))[:, order], threshold=0.5, margin=0.08
            )
        before, after = report["baseline_test"], report["candidate_test"]
        report["accepted"] = (
            after["raw_top1_accuracy"] >= before["raw_top1_accuracy"]
            and after["nll"] <= before["nll"] + gate["test_nll_maximum_increase"]
            and after["macro_f1_present_classes"] >= before["macro_f1_present_classes"]
        )
        report["test_status"] = "selected_model_evaluated"
        crop = cv2.imread(str(rows[int(val[0])]["path"]))
        for label, engine in [("baseline", InferenceBackend(spec)), ("candidate", backend)]:
            for _ in range(10):
                engine.predict(crop)
            durations = []
            for _ in range(100):
                start = time.perf_counter()
                engine.predict(crop)
                durations.append(time.perf_counter() - start)
            report[f"{label}_cpu_ms"] = float(np.mean(durations) * 1000)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
