"""Train a dropout-regularized residual classifier on frozen facial-expression features."""

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


def softmax(x):
    p = np.exp(x - x.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def forward(x, base, parameters, mean, std):
    a, b, c, d = parameters
    hidden = np.maximum(((x - mean) / std) @ a + b, 0)
    return base + hidden @ c + d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--feature-run", type=Path, help="Reuse verified frozen-CNN features")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--defer-test", action="store_true", help="Select using validation only")
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("epochs/patience must be positive")
    if not 1 <= args.hidden <= 1024 or not 0 <= args.dropout < 1 or not 0 < args.learning_rate < 1:
        parser.error("Invalid hidden size, dropout, or learning rate")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    spec = ModelSpec.load(args.baseline)
    model = onnx.load(spec.path)
    head = model.graph.node[-1]
    if head.op_type != "Gemm":
        raise ValueError("This trainer starts from the linear expression baseline")
    reference = json.loads(args.reference_report.read_text())
    if (
        hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        != reference["dataset_manifest_sha256"]
    ):
        raise ValueError("Reference dataset manifest changed")
    excluded = {str(Path(r["path"]).resolve()) for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if str(r["path"]) not in excluded]
    counts = {
        split: sum(r["split"] == split for r in rows) for split in ["train", "calibration", "test"]
    }
    if counts != reference["split_counts"]:
        raise ValueError("Reference split counts changed")
    params = {i.name: numpy_helper.to_array(i).copy() for i in model.graph.initializer}
    base_w = params[head.input[1]]
    base_b = params[head.input[2]]
    dim = base_w.shape[1]
    model.graph.output.append(
        helper.make_tensor_value_info(head.input[0], onnx.TensorProto.FLOAT, [1, dim])
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    n = len(rows)
    total = n + counts["train"]
    y = np.zeros(total, np.int64)
    splits = np.array([r["split"] for r in rows] + ["train"] * counts["train"])
    feature_sha = None
    if args.feature_run:
        cached = json.loads((args.feature_run / "training-report.json").read_text())
        if (
            cached["starting_model_sha256"] != spec.sha256
            or cached["manifest_sha256"] != reference["dataset_manifest_sha256"]
            or cached["split_counts"] != counts
        ):
            raise ValueError("Cached feature provenance mismatch")
        feature_path = args.feature_run / "features.npy"
        x = np.load(feature_path, mmap_mode="r")
        if x.shape != (total, dim) or x.dtype != np.float32:
            raise ValueError("Cached feature shape/dtype mismatch")
        with feature_path.open("rb") as handle:
            feature_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        y[:n] = [spec.labels.index(r["label"]) for r in rows]
        original_train = np.where(splits[:n] == "train")[0]
        y[n:] = y[original_train]
        checks = [int(original_train[0]), int(original_train[-1]), n, total - 1]
        checks += [i for i, row in enumerate(rows) if row["split"] == "calibration"][:2]
        for index in checks:
            source = index if index < n else int(original_train[index - n])
            image = cv2.imread(str(rows[source]["path"]))
            if image is None:
                raise ValueError("Cannot read cache verification image")
            if index >= n:
                image = cv2.flip(image, 1)
            actual = session.run(
                [head.input[0]], {session.get_inputs()[0].name: spec.tensor(image)}
            )[0][0]
            np.testing.assert_allclose(actual, x[index], atol=1e-5, rtol=1e-5)
        print("Feature cache provenance and inference checks passed", flush=True)
    else:
        x = np.lib.format.open_memmap(
            args.output / "features.npy", mode="w+", dtype="float32", shape=(total, dim)
        )
        flip = n
        for i, row in enumerate(rows):
            image = cv2.imread(str(row["path"]))
            if image is None:
                raise ValueError(f"Cannot decode {row['path']}")
            y[i] = spec.labels.index(row["label"])
            x[i] = session.run([head.input[0]], {session.get_inputs()[0].name: spec.tensor(image)})[
                0
            ][0]
            if row["split"] == "train":
                x[flip] = session.run(
                    [head.input[0]], {session.get_inputs()[0].name: spec.tensor(cv2.flip(image, 1))}
                )[0][0]
                y[flip] = y[i]
                flip += 1
            if (i + 1) % 500 == 0:
                print(f"Features: {i + 1}/{n} faces plus training flips", flush=True)
        x.flush()
    del session
    train = np.where(splits == "train")[0]
    val = np.where(splits == "calibration")[0]
    test = np.where(splits == "test")[0]
    mean = np.zeros(dim, np.float64)
    squares = np.zeros(dim, np.float64)
    for ids in np.array_split(train, 100):
        values = np.asarray(x[ids], np.float64)
        mean += values.sum(0)
        squares += (values * values).sum(0)
    mean = (mean / len(train)).astype(np.float32)
    std = np.maximum(
        np.sqrt(np.maximum(squares / len(train) - mean.astype(np.float64) ** 2, 0)), 0.1
    ).astype(np.float32)
    hidden = args.hidden
    parameters = [
        rng.normal(0, np.sqrt(2 / dim), (dim, hidden)).astype(np.float32),
        np.zeros(hidden, np.float32),
        np.zeros((hidden, 8), np.float32),
        np.zeros(8, np.float32),
    ]
    moments = [np.zeros_like(p) for p in parameters]
    variances = [np.zeros_like(p) for p in parameters]
    xv = np.asarray(x[val])
    base_val = xv @ base_w.T + base_b
    labels_val = y[val]

    def val_metrics(parameters):
        logits = forward(xv, base_val, parameters, mean, std)
        p = softmax(logits)
        return float(-np.log(np.maximum(p[np.arange(len(val)), labels_val], 1e-12)).mean()), float(
            (p.argmax(1) == labels_val).mean()
        )

    baseline_loss, baseline_accuracy = val_metrics(parameters)
    best_loss = baseline_loss
    best = [p.copy() for p in parameters]
    best_epoch = 0
    stale = 0
    step = 0
    history = []
    print("Baseline validation NLL/accuracy:", baseline_loss, baseline_accuracy, flush=True)
    for epoch in range(1, args.epochs + 1):
        for ids in np.array_split(rng.permutation(train), int(np.ceil(len(train) / 128))):
            original = np.asarray(x[ids])
            xb = (original - mean) / std
            a, b, c, d = parameters
            pre = xb @ a + b
            h = np.maximum(pre, 0)
            keep = 1 - args.dropout
            drop = (rng.random(h.shape) < keep).astype(np.float32) / keep
            hd = h * drop
            error = softmax(original @ base_w.T + base_b + hd @ c + d)
            error[np.arange(len(ids)), y[ids]] -= 1
            error /= len(ids)
            dh = (error @ c.T) * drop * (pre > 0)
            gradients = [xb.T @ dh + 0.0001 * a, dh.sum(0), hd.T @ error + 0.001 * c, error.sum(0)]
            norm = np.sqrt(sum(float((g * g).sum()) for g in gradients))
            scale = min(1.0, 5 / max(norm, 1e-9))
            step += 1
            for parameter, g, m, v in zip(parameters, gradients, moments, variances):
                g *= scale
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                parameter -= (
                    args.learning_rate
                    * (m / (1 - 0.9**step))
                    / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
        loss, accuracy = val_metrics(parameters)
        history.append({"epoch": epoch, "validation_nll": loss, "validation_top1": accuracy})
        print(f"Epoch {epoch}: validation NLL {loss:.4f}, top1 {accuracy:.4f}", flush=True)
        if loss < best_loss - 0.0001:
            best_loss = loss
            best = [p.copy() for p in parameters]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print("Validation plateau; stopping", flush=True)
            break
    a, b, c, d = best
    a = a / std[:, None]
    b = b - mean @ a
    candidate = onnx.load(spec.path)
    candidate.graph.node[-1].output[0] = "residual_baseline_logits"
    for name, value in [
        ("residual_w1", a),
        ("residual_b1", b),
        ("residual_w2", c),
        ("residual_b2", d),
    ]:
        candidate.graph.initializer.append(numpy_helper.from_array(value.astype(np.float32), name))
    candidate.graph.node.extend(
        [
            helper.make_node("MatMul", [head.input[0], "residual_w1"], ["residual_fc1"]),
            helper.make_node("Add", ["residual_fc1", "residual_b1"], ["residual_pre"]),
            helper.make_node("Relu", ["residual_pre"], ["residual_hidden"]),
            helper.make_node("MatMul", ["residual_hidden", "residual_w2"], ["residual_fc2"]),
            helper.make_node("Add", ["residual_fc2", "residual_b2"], ["residual_delta"]),
            helper.make_node(
                "Add", ["residual_baseline_logits", "residual_delta"], [head.output[0]]
            ),
        ]
    )
    onnx.checker.check_model(candidate)
    path = args.output / "expression-candidate.onnx"
    onnx.save(candidate, path)
    manifest = dataclasses.asdict(spec)
    manifest.update(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path = args.output / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    backend = InferenceBackend(ModelSpec.load(manifest_path))
    first = val[0]
    actual = backend.predict(cv2.imread(str(rows[first]["path"])))
    expected = softmax(
        forward(
            np.asarray(x[first : first + 1]),
            np.asarray(x[first : first + 1]) @ base_w.T + base_b,
            best,
            mean,
            std,
        )
    )[0]
    expected = expected[[spec.labels.index(label) for label in RAW_LABELS]]
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    baseline = trained = None
    if not args.defer_test:
        truth = np.array([RAW_LABELS.index(spec.labels[i]) for i in y[test]])

        def evaluate(logits):
            return metrics(
                truth,
                softmax(logits)[:, [spec.labels.index(label) for label in RAW_LABELS]],
                threshold=0.5,
                margin=0.08,
            )

        xt = np.asarray(x[test])
        bt = xt @ base_w.T + base_b
        baseline = evaluate(bt)
        trained = evaluate(forward(xt, bt, best, mean, std))
    vl, va = val_metrics(best)
    report = {
        "method": "Frozen CNN with residual ReLU classifier, training-only horizontal flips and dropout",
        "seed": args.seed,
        "hidden": hidden,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "test_deferred": args.defer_test,
        "feature_cache_sha256": feature_sha,
        "split_counts": counts,
        "augmented_training_examples": len(train),
        "starting_model_sha256": spec.sha256,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selected_epoch": best_epoch,
        "epochs_completed": len(history),
        "baseline_validation": {"nll": baseline_loss, "top1": baseline_accuracy},
        "candidate_validation": {"nll": vl, "top1": va},
        "baseline_test": baseline,
        "candidate_test": trained,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Same previously reported test set; no independent external benchmark.",
            "Identity separation and pretraining overlap are unknown.",
            "Visible expression labels do not establish internal feelings.",
        ],
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "baseline_top1": baseline["raw_top1_accuracy"] if baseline else None,
                "candidate_top1": trained["raw_top1_accuracy"] if trained else None,
                "selected_epoch": best_epoch,
                "candidate_validation": report["candidate_validation"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
