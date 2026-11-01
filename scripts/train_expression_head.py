"""Train the frozen HSEmotion CNN's final linear layer; preserve production weights."""

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
from onnx import numpy_helper

from expression_lab.backends import InferenceBackend, ModelSpec
from expression_lab.evaluation import metrics, read_manifest
from expression_lab.labels import RAW_LABELS
from expression_lab.recognition_models import hsemotion_spec


def softmax(x):
    p = np.exp(x - x.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/expressions/manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.output.exists():
        raise ValueError("Choose a new output directory; previous runs are preserved")
    args.output.mkdir(parents=True)
    started = time.time()
    rng = np.random.default_rng(42)
    spec = hsemotion_spec()
    spec.validate()
    graph = onnx.load(spec.path)
    head = graph.graph.node[-1]
    if head.op_type != "Gemm":
        raise ValueError("Pinned expression model must end with Gemm")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in head.attribute}
    if attrs.get("transB", 0) != 1 or attrs.get("alpha", 1) != 1 or attrs.get("beta", 1) != 1:
        raise ValueError("Unsupported classifier layout")
    initializers = {i.name: numpy_helper.to_array(i).copy() for i in graph.graph.initializer}
    w0, b0 = initializers[head.input[1]], initializers[head.input[2]]
    graph.graph.output.append(
        onnx.helper.make_tensor_value_info(head.input[0], onnx.TensorProto.FLOAT, [1, w0.shape[1]])
    )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        graph.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    rows = read_manifest(args.manifest)
    features, hashes, labels, splits, original_logits = [], [], [], [], []
    for idx, row in enumerate(rows):
        image = cv2.imread(str(row["path"]))
        if image is None:
            raise ValueError(f"Cannot decode {row['path']}")
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
        hashes.append(
            int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flat), 2)
        )
        logits, embedding = session.run(
            [head.output[0], head.input[0]], {session.get_inputs()[0].name: spec.tensor(image)}
        )
        features.append(embedding[0])
        original_logits.append(logits[0])
        labels.append(spec.labels.index(row["label"]))
        splits.append(row["split"])
        if (idx + 1) % 500 == 0:
            print(
                f"Extracted {idx + 1}/{len(rows)} faces in {time.time() - started:.1f}s", flush=True
            )
    x, y, splits = np.asarray(features), np.asarray(labels), np.asarray(splits)
    held = [h for h, s in zip(hashes, splits) if s == "test"]
    excluded = set()
    for i, (h, s) in enumerate(zip(hashes, splits)):
        if s == "calibration" and any((h ^ other).bit_count() <= 4 for other in held):
            excluded.add(i)
    held = [h for i, (h, s) in enumerate(zip(hashes, splits)) if s != "train" and i not in excluded]
    for i, (h, s) in enumerate(zip(hashes, splits)):
        if s == "train" and any((h ^ other).bit_count() <= 4 for other in held):
            excluded.add(i)
    masks = {
        s: np.array([sp == s and i not in excluded for i, sp in enumerate(splits)])
        for s in ["train", "calibration", "test"]
    }
    if any(m.sum() == 0 for m in masks.values()):
        raise ValueError("Each split needs accepted examples")
    train, val, test = (masks[s] for s in ["train", "calibration", "test"])
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0), 0.1)
    z = (x - mean) / std
    w = w0 * std
    b = b0 + mean @ w0.T
    anchor = w.copy()
    moments = [np.zeros_like(w), np.zeros_like(b)]
    variances = [np.zeros_like(w), np.zeros_like(b)]

    def loss(w, b):
        p = softmax(z[val] @ w.T + b)
        return float(-np.log(np.maximum(p[np.arange(val.sum()), y[val]], 1e-12)).mean())

    best_loss = loss(w, b)
    best = (w.copy(), b.copy())
    best_epoch = 0
    history = []
    step = 0
    indices = np.where(train)[0]
    for epoch in range(1, args.epochs + 1):
        for batch in np.array_split(
            rng.permutation(indices), max(1, int(np.ceil(len(indices) / 128)))
        ):
            xb, yb = z[batch], y[batch]
            error = softmax(xb @ w.T + b)
            error[np.arange(len(batch)), yb] -= 1
            gradients = [error.T @ xb / len(batch) + 0.005 * (w - anchor), error.mean(axis=0)]
            step += 1
            for parameter, gradient, m, v in zip([w, b], gradients, moments, variances):
                m *= 0.9
                m += 0.1 * gradient
                v *= 0.999
                v += 0.001 * gradient * gradient
                parameter -= (
                    0.0001 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
        current = loss(w, b)
        history.append({"epoch": epoch, "validation_nll": current})
        print(f"Epoch {epoch}/{args.epochs}: validation NLL {current:.4f}", flush=True)
        if current < best_loss:
            best_loss, best, best_epoch = current, (w.copy(), b.copy()), epoch
    w, b = best
    final_w = (w / std).astype(np.float32)
    final_b = (b - mean @ final_w.T).astype(np.float32)
    candidate = onnx.load(spec.path)
    for item in candidate.graph.initializer:
        if item.name == head.input[1]:
            item.CopyFrom(numpy_helper.from_array(final_w, item.name))
        if item.name == head.input[2]:
            item.CopyFrom(numpy_helper.from_array(final_b, item.name))
    onnx.checker.check_model(candidate)
    path = args.output / "expression-candidate.onnx"
    onnx.save(candidate, path)
    manifest = dataclasses.asdict(spec)
    manifest.update(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path = args.output / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    backend = InferenceBackend(ModelSpec.load(manifest_path))
    row = rows[int(np.where(test)[0][0])]
    exported = backend.predict(cv2.imread(str(row["path"])))
    predicted = softmax(x[test][:1] @ final_w.T + final_b)[0]
    canonical = np.array([predicted[spec.labels.index(label)] for label in RAW_LABELS])
    np.testing.assert_allclose(exported, canonical, atol=1e-5)
    truth = np.array([RAW_LABELS.index(spec.labels[label]) for label in y[test]])

    def evaluate(logits):
        p = softmax(logits)
        p = p[:, [spec.labels.index(label) for label in RAW_LABELS]]
        return metrics(truth, p, threshold=0.5, margin=0.08)

    baseline = evaluate(np.asarray(original_logits)[test])
    trained = evaluate(x[test] @ final_w.T + final_b)
    report = {
        "training_method": "Frozen HSEmotion CNN, Adam training of final classifier only",
        "seed": 42,
        "epochs_completed": args.epochs,
        "selected_epoch": best_epoch,
        "split_counts": {s: int(m.sum()) for s, m in masks.items()},
        "excluded_similarity_matches": [
            {"path": str(rows[i]["path"]), "split": rows[i]["split"]} for i in sorted(excluded)
        ],
        "validation_usage": "Checkpoint selection; do not treat as independent calibration data",
        "preprocessing": {"align": False, "tta": False, "input": "RGB224"},
        "baseline_test": baseline,
        "candidate_test": trained,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "production_model_changed": False,
        "limitations": [
            "Perceptual hashes do not establish person identity separation.",
            "Pretraining overlap with this dataset is unknown.",
            "Object detector is not trained by this script.",
            "Dataset licensing must be established separately.",
        ],
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_epoch": best_epoch,
                "baseline_accuracy": baseline["accuracy"],
                "candidate_accuracy": trained["accuracy"],
                "baseline_nll": baseline["nll"],
                "candidate_nll": trained["nll"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
