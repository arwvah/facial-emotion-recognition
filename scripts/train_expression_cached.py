"""Continue the existing linear expression head using verified, cached CNN features."""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from expression_lab.backends import InferenceBackend, ModelSpec
from expression_lab.evaluation import metrics, read_manifest
from expression_lab.labels import RAW_LABELS


def softmax(logits):
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("epochs and patience must be positive")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    started = time.time()
    rng = np.random.default_rng(42)
    spec = ModelSpec.load(args.baseline)
    reference = json.loads(args.reference_report.read_text())
    cache_report = json.loads((args.feature_run / "training-report.json").read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if (
        digest != reference["dataset_manifest_sha256"]
        or digest != cache_report["manifest_sha256"]
        or spec.sha256 != cache_report["starting_model_sha256"]
    ):
        raise ValueError("Feature cache model/dataset provenance does not match")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    counts = {
        split: sum(r["split"] == split for r in rows) for split in ["train", "calibration", "test"]
    }
    if counts != reference["split_counts"] or counts != cache_report["split_counts"]:
        raise ValueError("Split membership changed")
    graph = onnx.load(spec.path)
    head = graph.graph.node[-1]
    attrs = {a.name: helper.get_attribute_value(a) for a in head.attribute}
    if head.op_type != "Gemm" or attrs.get("transB", 0) != 1:
        raise ValueError("Expected the existing linear expression model")
    initializers = {i.name: numpy_helper.to_array(i).copy() for i in graph.graph.initializer}
    w0, b0 = initializers[head.input[1]], initializers[head.input[2]]
    feature_path = args.feature_run / "features.npy"
    x = np.load(feature_path, mmap_mode="r")
    if x.shape != (len(rows) + counts["train"], w0.shape[1]) or x.dtype != np.float32:
        raise ValueError("Feature cache shape/dtype mismatch")
    with feature_path.open("rb") as handle:
        feature_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    splits = np.array([r["split"] for r in rows] + ["train"] * counts["train"])
    original_train = np.where(splits[: len(rows)] == "train")[0]
    y = np.array(
        [spec.labels.index(r["label"]) for r in rows]
        + [spec.labels.index(rows[i]["label"]) for i in original_train]
    )
    train, val, test = (np.where(splits == s)[0] for s in ("train", "calibration", "test"))
    # Verify the cache against fresh CNN inference on train/validation views only.
    probe = onnx.load(spec.path)
    probe.graph.output.append(
        helper.make_tensor_value_info(head.input[0], onnx.TensorProto.FLOAT, [1, w0.shape[1]])
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        probe.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    checks = [int(original_train[i]) for i in (0, len(original_train) // 2, -1)]
    checks += [int(val[0]), int(val[-1]), len(rows), len(rows) + counts["train"] - 1]
    for index in checks:
        source_index = index if index < len(rows) else int(original_train[index - len(rows)])
        image = cv2.imread(str(rows[source_index]["path"]))
        if image is None:
            raise ValueError("Unreadable cache verification image")
        if index >= len(rows):
            image = cv2.flip(image, 1)
        actual = session.run([head.input[0]], {session.get_inputs()[0].name: spec.tensor(image)})[
            0
        ][0]
        np.testing.assert_allclose(actual, x[index], atol=1e-5, rtol=1e-5)
    del session, probe
    gate = {
        "validation_top1_minimum_gain": 0.003,
        "test_top1_must_not_decrease": True,
        "test_nll_maximum_increase": 0.02,
        "test_macro_f1_must_not_decrease": True,
    }
    (args.output / "promotion-gate.json").write_text(json.dumps(gate, indent=2))
    sums = np.zeros(w0.shape[1], np.float64)
    squares = np.zeros_like(sums)
    for ids in np.array_split(train, 100):
        values = np.asarray(x[ids], np.float64)
        sums += values.sum(0)
        squares += (values * values).sum(0)
    mean = (sums / len(train)).astype(np.float32)
    std = np.sqrt(np.maximum(squares / len(train) - mean.astype(np.float64) ** 2, 0))
    std = np.maximum(std, 0.1).astype(np.float32)
    w = w0 * std
    b = b0 + mean @ w0.T
    anchor = w.copy()
    moments = [np.zeros_like(w), np.zeros_like(b)]
    variances = [np.zeros_like(w), np.zeros_like(b)]
    xv = (x[val] - mean) / std

    def validation(w, b):
        p = softmax(xv @ w.T + b)
        return float(-np.log(np.maximum(p[np.arange(len(val)), y[val]], 1e-12)).mean()), float(
            (p.argmax(1) == y[val]).mean()
        )

    baseline_loss, baseline_accuracy = validation(w, b)
    best_loss, best_epoch = baseline_loss, 0
    best, history = (w.copy(), b.copy()), []
    stale = step = 0
    print(f"Verified cache; baseline validation: {baseline_accuracy:.4%}", flush=True)
    for epoch in range(1, args.epochs + 1):
        for ids in np.array_split(rng.permutation(train), int(np.ceil(len(train) / 256))):
            xb = (x[ids] - mean) / std
            xb *= (rng.random(xb.shape) < 0.9).astype(np.float32) / 0.9
            error = softmax(xb @ w.T + b)
            error -= 0.02 / len(spec.labels)
            error[np.arange(len(ids)), y[ids]] -= 0.98
            gradients = [error.T @ xb / len(ids) + 0.002 * (w - anchor), error.mean(0)]
            step += 1
            for parameter, g, m, v in zip([w, b], gradients, moments, variances):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                parameter -= (
                    0.00005 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
        loss, accuracy = validation(w, b)
        history.append({"epoch": epoch, "validation_nll": loss, "validation_top1": accuracy})
        print(
            f"Epoch {epoch}/{args.epochs}: validation NLL {loss:.4f}, top1 {accuracy:.4%}",
            flush=True,
        )
        if loss < best_loss - 0.0001:
            best_loss, best, best_epoch, stale = loss, (w.copy(), b.copy()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            print("Validation plateau; stopping", flush=True)
            break
    order = [spec.labels.index(label) for label in RAW_LABELS]

    def export(weights, bias, filename, manifest_name):
        candidate = onnx.load(spec.path)
        for item in candidate.graph.initializer:
            if item.name == head.input[1]:
                item.CopyFrom(numpy_helper.from_array(weights, item.name))
            elif item.name == head.input[2]:
                item.CopyFrom(numpy_helper.from_array(bias, item.name))
            else:
                np.testing.assert_array_equal(numpy_helper.to_array(item), initializers[item.name])
        onnx.checker.check_model(candidate)
        model_path = args.output / filename
        onnx.save(candidate, model_path)
        manifest = dataclasses.asdict(spec)
        manifest.update(
            path=model_path.name, sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()
        )
        manifest_path = args.output / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2))
        backend = InferenceBackend(ModelSpec.load(manifest_path))
        actual = backend.predict(cv2.imread(str(rows[int(val[0])]["path"])))
        expected = softmax(x[val[:1]] @ weights.T + bias)[0]
        np.testing.assert_allclose(actual, expected[order], atol=1e-5)

    # Preserve the actual last trained head even when the starting checkpoint wins selection.
    last_loss, last_accuracy = validation(w, b)
    last_w = (w / std).astype(np.float32)
    last_b = (b - mean @ last_w.T).astype(np.float32)
    export(last_w, last_b, "last-trained.onnx", "last-trained.json")
    w, b = best
    final_w = w0 if best_epoch == 0 else (w / std).astype(np.float32)
    final_b = b0 if best_epoch == 0 else (b - mean @ final_w.T).astype(np.float32)
    export(final_w, final_b, "expression-candidate.onnx", "model.json")
    # Read test features for scoring only after checkpoint selection and export checks.
    truth = np.array([RAW_LABELS.index(spec.labels[i]) for i in y[test]])

    def evaluate(w, b):
        return metrics(truth, softmax(x[test] @ w.T + b)[:, order], threshold=0.5, margin=0.08)

    baseline, trained = evaluate(w0, b0), evaluate(final_w, final_b)
    vl, va = validation(w, b)
    accepted = (
        va >= baseline_accuracy + gate["validation_top1_minimum_gain"]
        and trained["raw_top1_accuracy"] >= baseline["raw_top1_accuracy"]
        and trained["nll"] <= baseline["nll"] + gate["test_nll_maximum_increase"]
        and trained["macro_f1_present_classes"] >= baseline["macro_f1_present_classes"]
    )
    report = {
        "method": "Existing frozen CNN, continued linear head training, training-only flips, 10% feature dropout and 2% label smoothing",
        "starting_model_sha256": spec.sha256,
        "dataset_manifest_sha256": digest,
        "feature_cache_sha256": feature_sha,
        "cache_verification_views": len(checks),
        "seed": 42,
        "split_counts": counts,
        "training_views": len(train),
        "selected_epoch": best_epoch,
        "epochs_completed": len(history),
        "baseline_validation": {"nll": baseline_loss, "top1": baseline_accuracy},
        "candidate_validation": {"nll": vl, "top1": va},
        "last_trained_validation": {"nll": last_loss, "top1": last_accuracy},
        "selected_checkpoint_is_starting_baseline": best_epoch == 0,
        "baseline_test": baseline,
        "candidate_test": trained,
        "promotion_gate": gate,
        "accepted": accepted,
        "production_model_changed": False,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Previously reported test subset, not an independent external benchmark",
            "Validation is used for checkpoint selection, not independent calibration",
            "Subject separation and pretraining overlap unknown",
            "Visible expression labels do not establish internal feelings",
        ],
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "accepted": accepted,
                "baseline_top1": baseline["raw_top1_accuracy"],
                "candidate_top1": trained["raw_top1_accuracy"],
                "selected_epoch": best_epoch,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
