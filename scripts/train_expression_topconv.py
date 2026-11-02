"""Fine-tune the existing CNN's final pointwise convolution and classifier using NumPy."""

import argparse
import copy
import dataclasses
import hashlib
import json
import time
from collections import Counter
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
    p = np.exp(x - x.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def forward(features, parameters):
    w, b, h, c = parameters
    z = features @ w.T + b
    s = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    pooled = (z * s).mean(1)
    return pooled @ h.T + c, (z, s, pooled)


def gradients(features, labels, parameters):
    logits, (z, s, pooled) = forward(features, parameters)
    error = softmax(logits)
    error[np.arange(len(labels)), labels] -= 1
    error /= len(labels)
    dz = (error @ parameters[2])[:, None, :] / features.shape[1]
    dz = dz * s * (1 + z * (1 - s))
    return [
        dz.reshape(-1, dz.shape[-1]).T @ features.reshape(-1, features.shape[-1]),
        dz.sum((0, 1)),
        error.T @ pooled,
        error.sum(0),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if min(args.per_class, args.epochs, args.patience) < 1:
        parser.error("Counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    spec = ModelSpec.load(args.baseline)
    if spec.color != "rgb" or (spec.width, spec.height) != (224, 224) or len(spec.labels) != 8:
        raise ValueError("Expected the existing eight-class RGB224 HSEmotion model")
    model = onnx.load(spec.path)
    tail = list(model.graph.node[-6:])
    if [n.op_type for n in tail] != [
        "Conv",
        "Sigmoid",
        "Mul",
        "GlobalAveragePool",
        "Flatten",
        "Gemm",
    ]:
        raise ValueError("Unsupported final convolution/classifier layout")
    conv, _, _, _, _, head = tail
    head_attrs = {a.name: helper.get_attribute_value(a) for a in head.attribute}
    if (
        head_attrs.get("transB", 0) != 1
        or head_attrs.get("alpha", 1) != 1
        or head_attrs.get("beta", 1) != 1
    ):
        raise ValueError("Unsupported classifier layout")
    if (
        list(tail[1].input) != [conv.output[0]]
        or set(tail[2].input) != {conv.output[0], tail[1].output[0]}
        or list(tail[3].input) != [tail[2].output[0]]
        or list(tail[4].input) != [tail[3].output[0]]
        or head.input[0] != tail[4].output[0]
    ):
        raise ValueError("Expected Conv -> SiLU -> average pool -> classifier connections")
    attrs = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
    if any(
        attrs.get(k) != v
        for k, v in {
            "kernel_shape": [1, 1],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "group": 1,
        }.items()
    ):
        raise ValueError("Expected a pointwise, ungrouped final convolution")
    initial = {i.name: numpy_helper.to_array(i).copy() for i in model.graph.initializer}
    parameters = [
        initial[conv.input[1]][:, :, 0, 0].copy(),
        initial[conv.input[2]].copy(),
        initial[head.input[1]].copy(),
        initial[head.input[2]].copy(),
    ]
    anchor = [p.copy() for p in parameters]
    reference = json.loads(args.reference_report.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if digest != reference["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "calibration", "test")}
    if counts != reference["split_counts"]:
        raise ValueError("Split membership changed")
    rng = np.random.default_rng(42)
    train = []
    for name in spec.labels:
        ids = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == name]
        train.extend(rng.permutation(ids)[: args.per_class].tolist())
    val = [i for i, r in enumerate(rows) if r["split"] == "calibration"]
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    selected_rows = [rows[i] for i in train + val]
    labels = np.array([spec.labels.index(r["label"]) for r in selected_rows])
    ntrain = len(train)
    plan = {
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "seed": 42,
        "training_subset_counts": dict(Counter(rows[i]["label"] for i in train)),
        "training_images": ntrain,
        "validation_images": len(val),
        "learning_rate": 0.00001,
        "batch_size": 16,
        "anchor_penalty": 0.0001,
        "trainable": "Final pointwise convolution weights/bias and classifier weights/bias; all earlier layers frozen",
        "selection": "Minimum validation NLL, including starting checkpoint",
        "promotion_gate": {
            "validation_top1_minimum_gain": 0.003,
            "test_top1_must_not_decrease": True,
            "test_macro_f1_must_not_decrease": True,
            "test_nll_maximum_increase": 0.02,
        },
    }
    (args.output / "experiment-plan.json").write_text(json.dumps(plan, indent=2))
    probe = copy.deepcopy(model)
    del probe.graph.node[-6:]
    del probe.graph.output[:]
    probe.graph.output.append(
        helper.make_tensor_value_info(
            conv.input[0], onnx.TensorProto.FLOAT, [1, parameters[0].shape[1], 7, 7]
        )
    )
    used = {name for node in probe.graph.node for name in node.input}
    retained = [i for i in probe.graph.initializer if i.name in used]
    del probe.graph.initializer[:]
    probe.graph.initializer.extend(retained)
    onnx.checker.check_model(probe)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        probe.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    x = np.lib.format.open_memmap(
        args.output / "spatial-features.npy",
        mode="w+",
        dtype="float32",
        shape=(len(selected_rows), 49, parameters[0].shape[1]),
    )
    original_backend = InferenceBackend(spec)
    for i, row in enumerate(selected_rows):
        image = cv2.imread(str(row["path"]))
        if image is None:
            raise ValueError("Cannot decode training/validation image")
        tensor = spec.tensor(image)
        a = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        x[i] = a[0].reshape(parameters[0].shape[1], 49).T
        if i in (0, ntrain, len(selected_rows) - 1):
            expected = softmax(forward(x[i : i + 1], parameters)[0])[0]
            actual = original_backend.predict(image)
            np.testing.assert_allclose(
                actual, expected[[spec.labels.index(s) for s in RAW_LABELS]], atol=1e-5
            )
        if (i + 1) % 500 == 0:
            print(
                f"Spatial features {i + 1}/{len(selected_rows)}; {time.time() - started:.0f}s",
                flush=True,
            )
    x.flush()
    del session, probe, original_backend

    def validation(parameters):
        probabilities = []
        for indices in np.array_split(np.arange(ntrain, len(x)), int(np.ceil(len(val) / 32))):
            probabilities.append(softmax(forward(np.asarray(x[indices]), parameters)[0]))
        p = np.concatenate(probabilities)
        truth = labels[ntrain:]
        return float(-np.log(np.maximum(p[np.arange(len(p)), truth], 1e-12)).mean()), float(
            (p.argmax(1) == truth).mean()
        )

    baseline_loss, baseline_accuracy = validation(parameters)
    best_loss, best_epoch, best = baseline_loss, 0, [p.copy() for p in parameters]
    stale = step = 0
    history = []
    moments = [np.zeros_like(p) for p in parameters]
    variances = [np.zeros_like(p) for p in parameters]
    for epoch in range(1, args.epochs + 1):
        for indices in np.array_split(rng.permutation(ntrain), int(np.ceil(ntrain / 16))):
            gs = gradients(np.asarray(x[indices]), labels[indices], parameters)
            gs = [g + 0.0001 * (p - a) for g, p, a in zip(gs, parameters, anchor)]
            norm = np.sqrt(sum(float((g * g).sum()) for g in gs))
            scale = min(1.0, 5 / max(norm, 1e-9))
            step += 1
            for p, g, m, v in zip(parameters, gs, moments, variances):
                g *= scale
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                p -= 0.00001 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        loss, accuracy = validation(parameters)
        history.append({"epoch": epoch, "validation_nll": loss, "validation_top1": accuracy})
        print(
            f"Epoch {epoch}: validation NLL {loss:.4f}, top1 {accuracy:.4%}, elapsed {time.time() - started:.0f}s",
            flush=True,
        )
        if loss < best_loss - 0.0001:
            best_loss, best_epoch, best, stale = loss, epoch, [p.copy() for p in parameters], 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    np.savez(args.output / "last-parameters.npz", **{f"p{i}": p for i, p in enumerate(parameters)})
    replacements = {
        conv.input[1]: best[0][:, :, None, None],
        conv.input[2]: best[1],
        head.input[1]: best[2],
        head.input[2]: best[3],
    }
    candidate = onnx.load(spec.path)
    for item in candidate.graph.initializer:
        if item.name in replacements:
            item.CopyFrom(numpy_helper.from_array(replacements[item.name], item.name))
        else:
            np.testing.assert_array_equal(numpy_helper.to_array(item), initial[item.name])
    onnx.checker.check_model(candidate)
    path = args.output / "expression-topconv.onnx"
    onnx.save(candidate, path)
    manifest = dataclasses.asdict(spec)
    manifest.update(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.output / "model.json").write_text(json.dumps(manifest, indent=2))
    backend = InferenceBackend(ModelSpec.load(args.output / "model.json"))
    order = [spec.labels.index(s) for s in RAW_LABELS]
    probabilities = []
    for position, row in enumerate(selected_rows[ntrain:]):
        image = cv2.imread(str(row["path"]))
        p = backend.predict(image)
        probabilities.append(p)
        if position in (0, len(val) // 2, len(val) - 1):
            expected = softmax(forward(x[ntrain + position : ntrain + position + 1], best)[0])[0][
                order
            ]
            np.testing.assert_allclose(p, expected, atol=1e-5)
    truth = np.array([RAW_LABELS.index(r["label"]) for r in selected_rows[ntrain:]])
    actual_val = metrics(truth, probabilities, threshold=0.5, margin=0.08)
    report = {
        "plan": plan,
        "starting_model_sha256": spec.sha256,
        "model_sha256": manifest["sha256"],
        "manifest_sha256": digest,
        "split_counts": counts,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "baseline_validation": {"nll": baseline_loss, "top1": baseline_accuracy},
        "candidate_validation": actual_val,
        "accepted": False,
        "production_model_changed": False,
        "test_status": "not_evaluated_validation_gate_failed",
        "history": history,
        "limitations": [
            "Repeatedly reported validation/test subsets; no independent generalization claim",
            "Earlier CNN layers frozen; no full-backbone training",
            "Balanced subset of original training images; no additional dataset used by this candidate",
        ],
    }
    if actual_val["raw_top1_accuracy"] >= baseline_accuracy + 0.003:
        scores = {}
        for name, engine in [("baseline", InferenceBackend(spec)), ("candidate", backend)]:
            ps = []
            durations = []
            for i in test:
                image = cv2.imread(str(rows[i]["path"]))
                start = time.perf_counter()
                ps.append(engine.predict(image))
                durations.append(time.perf_counter() - start)
            scores[name] = metrics(
                [RAW_LABELS.index(rows[i]["label"]) for i in test], ps, threshold=0.5, margin=0.08
            )
            report[f"{name}_cpu_ms"] = float(np.mean(durations) * 1000)
        before, after = scores["baseline"], scores["candidate"]
        report.update(
            baseline_test=before, candidate_test=after, test_status="selected_model_evaluated"
        )
        report["accepted"] = (
            after["raw_top1_accuracy"] >= before["raw_top1_accuracy"]
            and after["macro_f1_present_classes"] >= before["macro_f1_present_classes"]
            and after["nll"] <= before["nll"] + 0.02
        )
    report["elapsed_seconds"] = time.time() - started
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
