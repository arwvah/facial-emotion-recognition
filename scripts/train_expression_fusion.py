"""Train a fused expression head using the two already installed, frozen CNNs."""

import argparse
import copy
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
from onnx.compose import add_prefix

from expression_lab.backends import InferenceBackend, ModelSpec
from expression_lab.emotion import MODEL_PATH, MODEL_SHA256
from expression_lab.evaluation import metrics, read_manifest
from expression_lab.labels import RAW_LABELS


def softmax(logits):
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fer_branch(spec, baseline):
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Installed FER+ checksum mismatch")
    source = onnx.load(MODEL_PATH)
    final = source.graph.node[-2]
    if final.op_type != "MatMul" or source.graph.node[-1].op_type != "Add":
        raise ValueError("Unsupported FER+ classifier layout")
    weights = {i.name: numpy_helper.to_array(i) for i in source.graph.initializer}
    dim = weights[final.input[1]].shape[0]
    source = add_prefix(source, "fusion_fer_")
    feature = source.graph.node[-2].input[0]
    del source.graph.node[-2:]
    for node in source.graph.node:
        if node.op_type == "Dropout":
            node.op_type = "Identity"
            del node.attribute[:]
    used = {s for node in source.graph.node for s in node.input}
    initializers = [i for i in source.graph.initializer if i.name in used]
    mean = np.broadcast_to(spec.mean, (3,))
    std = np.broadcast_to(spec.std, (3,))
    luminance = np.array([0.299, 0.587, 0.114], np.float32)
    kernel = (luminance * std / spec.scale).astype(np.float32).reshape(1, 3, 1, 1)
    bias = np.array([float(luminance @ mean) / spec.scale], np.float32)
    initializers += [
        numpy_helper.from_array(kernel, "fusion_gray_kernel"),
        numpy_helper.from_array(bias, "fusion_gray_bias"),
        numpy_helper.from_array(np.array([1, 1, 64, 64], np.int64), "fusion_gray_sizes"),
        numpy_helper.from_array(np.array([], np.float32), "fusion_resize_empty"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            [baseline.graph.input[0].name, "fusion_gray_kernel", "fusion_gray_bias"],
            ["fusion_gray224"],
        ),
        helper.make_node(
            "Resize",
            ["fusion_gray224", "fusion_resize_empty", "fusion_resize_empty", "fusion_gray_sizes"],
            [source.graph.input[0].name],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        ),
        *source.graph.node,
    ]
    graph = helper.make_graph(
        nodes,
        "local-fer-feature-branch",
        [copy.deepcopy(baseline.graph.input[0])],
        [helper.make_tensor_value_info(feature, onnx.TensorProto.FLOAT, [1, dim])],
        initializers,
    )
    branch = helper.make_model(graph, opset_imports=list(baseline.opset_import))
    branch.ir_version = baseline.ir_version
    onnx.checker.check_model(branch)
    return branch, feature, dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--fer-feature-cache", type=Path)
    parser.add_argument("--extra-manifest", type=Path)
    parser.add_argument("--defer-test", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("epochs and patience must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rng = np.random.default_rng(42)
    spec = ModelSpec.load(args.baseline)
    if spec.color != "rgb" or (spec.width, spec.height) != (224, 224):
        raise ValueError("Expected the installed RGB224 baseline")
    baseline = onnx.load(spec.path)
    head = baseline.graph.node[-1]
    if head.op_type != "Gemm":
        raise ValueError("Expected the existing linear baseline")
    params = {i.name: numpy_helper.to_array(i).copy() for i in baseline.graph.initializer}
    w0, b0 = params[head.input[1]], params[head.input[2]]
    reference = json.loads(args.reference_report.read_text())
    cached = json.loads((args.feature_run / "training-report.json").read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if (
        digest != reference["dataset_manifest_sha256"]
        or digest != cached["manifest_sha256"]
        or spec.sha256 != cached["starting_model_sha256"]
    ):
        raise ValueError("Feature cache provenance mismatch")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "calibration", "test")}
    if counts != reference["split_counts"] or counts != cached["split_counts"]:
        raise ValueError("Split counts changed")
    n, total = len(rows), len(rows) + counts["train"]
    feature_path = args.feature_run / "features.npy"
    x = np.load(feature_path, mmap_mode="r")
    if x.shape != (total, w0.shape[1]) or x.dtype != np.float32:
        raise ValueError("Feature cache shape/dtype mismatch")
    with feature_path.open("rb") as handle:
        feature_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    splits = np.array([r["split"] for r in rows] + ["train"] * counts["train"])
    original_train = np.where(splits[:n] == "train")[0]
    y = np.array([spec.labels.index(r["label"]) for r in rows])
    y = np.concatenate([y, y[original_train]])
    train, val, test = (np.where(splits == s)[0] for s in ("train", "calibration", "test"))
    branch, fer_feature, fer_dim = fer_branch(spec, baseline)
    onnx.save(branch, args.output / "fer-feature-branch.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    session = ort.InferenceSession(
        branch.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    probe = onnx.load(spec.path)
    probe.graph.output.append(
        helper.make_tensor_value_info(head.input[0], onnx.TensorProto.FLOAT, [1, w0.shape[1]])
    )
    verifier = ort.InferenceSession(
        probe.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for index in [int(original_train[0]), int(original_train[-1]), int(val[0]), int(val[-1])]:
        image = cv2.imread(str(rows[index]["path"]))
        actual = verifier.run([head.input[0]], {verifier.get_inputs()[0].name: spec.tensor(image)})[
            0
        ][0]
        np.testing.assert_allclose(actual, x[index], atol=1e-5, rtol=1e-5)
    del probe
    gate = {
        "validation_top1_minimum_gain": 0.003,
        "test_top1_must_not_decrease": True,
        "test_nll_maximum_increase": 0.02,
        "test_macro_f1_must_not_decrease": True,
    }
    (args.output / "promotion-gate.json").write_text(json.dumps(gate, indent=2))
    if args.fer_feature_cache:
        old = json.loads((args.fer_feature_cache / "training-report.json").read_text())
        if (
            old["starting_model_sha256"] != spec.sha256
            or old["ferplus_model_sha256"] != MODEL_SHA256
            or old["manifest_sha256"] != digest
            or old["test_status"] != "selected_model_evaluated"
        ):
            raise ValueError("FER+ feature cache provenance/completeness mismatch")
        xf = np.load(args.fer_feature_cache / "fer-features.npy", mmap_mode="r")
        if xf.shape != (total, fer_dim) or xf.dtype != np.float32:
            raise ValueError("FER+ feature cache shape mismatch")
        for index in [int(original_train[0]), int(val[0]), n, total - 1]:
            source = index if index < n else int(original_train[index - n])
            image = cv2.imread(str(rows[source]["path"]))
            if index >= n:
                image = cv2.flip(image, 1)
            actual = session.run(None, {session.get_inputs()[0].name: spec.tensor(image)})[0][0]
            np.testing.assert_allclose(actual, xf[index], atol=1e-5, rtol=1e-5)
        print("Reused FER+ features; provenance and inference checks passed", flush=True)
    else:
        xf = np.lib.format.open_memmap(
            args.output / "fer-features.npy", mode="w+", dtype="float32", shape=(total, fer_dim)
        )
        for completed, index in enumerate(np.concatenate([train, val]), start=1):
            source = int(index) if index < n else int(original_train[index - n])
            image = cv2.imread(str(rows[source]["path"]))
            if image is None:
                raise ValueError("Cannot decode training/validation image")
            if index >= n:
                image = cv2.flip(image, 1)
            xf[index] = session.run(None, {session.get_inputs()[0].name: spec.tensor(image)})[0][0]
            if completed % 1000 == 0:
                print(f"FER+ features {completed}/{len(train) + len(val)}", flush=True)
        xf.flush()
    extra = read_manifest(args.extra_manifest) if args.extra_manifest else []
    extra_x = None
    if extra:
        if any(r["split"] != "train" for r in extra):
            raise ValueError("Additional examples must be training-only")
        extra_x = np.lib.format.open_memmap(
            args.output / "extra-features.npy",
            mode="w+",
            dtype="float32",
            shape=(2 * len(extra), w0.shape[1] + fer_dim),
        )
        for index in range(2 * len(extra)):
            image = cv2.imread(str(extra[index % len(extra)]["path"]))
            if image is None:
                raise ValueError("Cannot decode extra training image")
            if index >= len(extra):
                image = cv2.flip(image, 1)
            tensor = spec.tensor(image)
            hs = verifier.run([head.input[0]], {verifier.get_inputs()[0].name: tensor})[0][0]
            fer = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
            extra_x[index] = np.concatenate([hs, fer])
            if (index + 1) % 500 == 0:
                print(f"Extra training features {index + 1}/{len(extra_x)}", flush=True)
        extra_x.flush()
        train = np.concatenate([train, np.arange(total, total + len(extra_x))])
        labels = np.array([spec.labels.index(r["label"]) for r in extra])
        y = np.concatenate([y, labels, labels])
    del verifier

    def features(indices):
        if extra_x is None or np.all(indices < total):
            return np.concatenate([x[indices], xf[indices]], axis=1)
        result = np.empty((len(indices), w0.shape[1] + fer_dim), np.float32)
        original_indices = indices < total
        source = indices[original_indices]
        result[original_indices] = np.concatenate([x[source], xf[source]], axis=1)
        result[~original_indices] = extra_x[indices[~original_indices] - total]
        return result

    dim = w0.shape[1] + fer_dim
    mean = np.zeros(dim, np.float64)
    squares = np.zeros(dim, np.float64)
    for indices in np.array_split(train, 100):
        values = features(indices).astype(np.float64)
        mean += values.sum(0)
        squares += (values * values).sum(0)
    mean = (mean / len(train)).astype(np.float32)
    std = np.sqrt(np.maximum(squares / len(train) - mean.astype(np.float64) ** 2, 0))
    std = np.maximum(std, 0.1).astype(np.float32)
    initial = np.concatenate([w0, np.zeros((len(spec.labels), fer_dim), np.float32)], axis=1)
    w = initial * std
    b = b0 + mean @ initial.T
    anchor = w.copy()
    moments = [np.zeros_like(w), np.zeros_like(b)]
    variances = [np.zeros_like(w), np.zeros_like(b)]
    xv = (features(val) - mean) / std

    def validation(w, b):
        p = softmax(xv @ w.T + b)
        return float(-np.log(np.maximum(p[np.arange(len(val)), y[val]], 1e-12)).mean()), float(
            (p.argmax(1) == y[val]).mean()
        )

    baseline_loss, baseline_accuracy = validation(w, b)
    best_loss, best_epoch = baseline_loss, 0
    best, history = (w.copy(), b.copy()), []
    stale = step = 0
    print(f"Fusion baseline validation: {baseline_accuracy:.4%}", flush=True)
    for epoch in range(1, args.epochs + 1):
        for indices in np.array_split(rng.permutation(train), int(np.ceil(len(train) / 256))):
            xb = (features(indices) - mean) / std
            error = softmax(xb @ w.T + b)
            error[np.arange(len(indices)), y[indices]] -= 1
            gradients = [error.T @ xb / len(indices) + 0.01 * (w - anchor), error.mean(0)]
            step += 1
            for parameter, g, m, v in zip([w, b], gradients, moments, variances):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                parameter -= (
                    0.0001 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
        loss, accuracy = validation(w, b)
        history.append({"epoch": epoch, "validation_nll": loss, "validation_top1": accuracy})
        print(f"Epoch {epoch}: validation NLL {loss:.4f}, top1 {accuracy:.4%}", flush=True)
        if loss < best_loss - 0.0001:
            best_loss, best, best_epoch, stale = loss, (w.copy(), b.copy()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            print("Validation plateau; stopping", flush=True)
            break
    w, b = best
    final_w = (w / std).astype(np.float32)
    final_b = (b - mean @ final_w.T).astype(np.float32)
    candidate = onnx.load(spec.path)
    del candidate.graph.node[-1]
    retained = [i for i in candidate.graph.initializer if i.name not in head.input[1:]]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(retained)
    candidate.graph.node.extend(branch.graph.node)
    candidate.graph.initializer.extend(branch.graph.initializer)
    candidate.graph.initializer.extend(
        [
            numpy_helper.from_array(final_w, "fusion_head_weight"),
            numpy_helper.from_array(final_b, "fusion_head_bias"),
        ]
    )
    candidate.graph.node.extend(
        [
            helper.make_node("Concat", [head.input[0], fer_feature], ["fusion_features"], axis=1),
            helper.make_node(
                "Gemm",
                ["fusion_features", "fusion_head_weight", "fusion_head_bias"],
                [head.output[0]],
                transB=1,
            ),
        ]
    )
    all_weights = {i.name: numpy_helper.to_array(i) for i in candidate.graph.initializer}
    for name, value in params.items():
        if name not in head.input[1:]:
            np.testing.assert_array_equal(all_weights[name], value)
    for item in branch.graph.initializer:
        np.testing.assert_array_equal(all_weights[item.name], numpy_helper.to_array(item))
    onnx.checker.check_model(candidate)
    model_path = args.output / "expression-fusion.onnx"
    onnx.save(candidate, model_path)
    manifest = dataclasses.asdict(spec)
    manifest.update(
        path=model_path.name, sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    manifest_path = args.output / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    backend = InferenceBackend(ModelSpec.load(manifest_path))
    order = [spec.labels.index(label) for label in RAW_LABELS]
    for index in (0, len(val) // 2, len(val) - 1):
        image = cv2.imread(str(rows[int(val[index])]["path"]))
        actual = backend.predict(image)
        expected = softmax(features(val[index : index + 1]) @ final_w.T + final_b)[0][order]
        np.testing.assert_allclose(actual, expected, atol=1e-5)
    vl, va = validation(w, b)
    report = {
        "method": "Frozen HSEmotion and FER+ CNN feature fusion, trained linear classifier",
        "starting_model_sha256": spec.sha256,
        "candidate_model_sha256": manifest["sha256"],
        "starting_model_contract": spec.signature(),
        "ferplus_model_sha256": MODEL_SHA256,
        "manifest_sha256": digest,
        "hsemotion_feature_cache_sha256": feature_sha,
        "split_counts": counts,
        "training_views": len(train),
        "extra_training_images": len(extra),
        "extra_manifest_sha256": hashlib.sha256(args.extra_manifest.read_bytes()).hexdigest()
        if args.extra_manifest
        else None,
        "seed": 42,
        "test_deferred": args.defer_test,
        "selected_epoch": best_epoch,
        "epochs_completed": len(history),
        "baseline_validation": {"nll": baseline_loss, "top1": baseline_accuracy},
        "candidate_validation": {"nll": vl, "top1": va},
        "promotion_gate": gate,
        "accepted": False,
        "production_model_changed": False,
        "test_status": "deferred" if args.defer_test else "not_evaluated_validation_gate_failed",
        "history": history,
        "limitations": [
            "Previously reported validation/test subsets; not an independent external benchmark",
            "Subject separation, dataset rights and pretraining overlap unverified",
            "New FER+ branch uses grayscale bilinear64 from the common RGB224 tensor",
            "Visible expression estimates do not establish feelings",
        ],
    }
    if (
        not args.defer_test
        and best_epoch > 0
        and va >= baseline_accuracy + gate["validation_top1_minimum_gain"]
    ):
        if not args.fer_feature_cache:
            for index in test:
                image = cv2.imread(str(rows[int(index)]["path"]))
                xf[index] = session.run(None, {session.get_inputs()[0].name: spec.tensor(image)})[
                    0
                ][0]
            xf.flush()
        truth = np.array([RAW_LABELS.index(spec.labels[i]) for i in y[test]])
        p0 = softmax(x[test] @ w0.T + b0)[:, order]
        p1 = softmax(features(test) @ final_w.T + final_b)[:, order]
        before = metrics(truth, p0, threshold=0.5, margin=0.08)
        after = metrics(truth, p1, threshold=0.5, margin=0.08)
        report.update(
            baseline_test=before, candidate_test=after, test_status="selected_model_evaluated"
        )
        report["accepted"] = (
            after["raw_top1_accuracy"] >= before["raw_top1_accuracy"]
            and after["nll"] <= before["nll"] + gate["test_nll_maximum_increase"]
            and after["macro_f1_present_classes"] >= before["macro_f1_present_classes"]
        )
        image = cv2.imread(str(rows[int(val[0])]["path"]))
        for name, engine in [("baseline", InferenceBackend(spec)), ("candidate", backend)]:
            for _ in range(10):
                engine.predict(image)
            start = time.perf_counter()
            for _ in range(100):
                engine.predict(image)
            report[f"{name}_cpu_ms"] = (time.perf_counter() - start) * 10
    report["elapsed_seconds"] = time.time() - started
    with Path(xf.filename).open("rb") as handle:
        report["fer_feature_cache_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
