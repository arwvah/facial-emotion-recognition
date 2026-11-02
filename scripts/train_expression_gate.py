"""Fit a source-domain selector for the existing color and grayscale expression heads."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper

from expression_lab.backends import InferenceBackend, ModelSpec
from expression_lab.emotion import MODEL_PATH, MODEL_SHA256
from expression_lab.evaluation import metrics, read_manifest
from expression_lab.labels import RAW_LABELS


def softmax(x):
    p = np.exp(x - x.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-run", type=Path, required=True)
    parser.add_argument("--extra-run", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--extra-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "seed": 13,
        "domain_epochs_maximum": 100,
        "domain_patience": 10,
        "blend_strengths": [0.0, 0.25, 0.5, 1.0],
        "selection": "Minimize equally weighted color/FER-training-heldout expression NLL, constrained to color validation top-1 decrease <=0.001 and NLL increase <=0.005 against fusion. Domain model fitted only on training source labels. No test scores used for selection.",
        "primary_deployment_gate": "Same original gate: validation gain >=0.003, test raw top-1 and macro-F1 nondecreasing, test NLL increase <=0.02 against original baseline.",
    }
    (args.output / "experiment-plan.json").write_text(json.dumps(plan, indent=2))
    spec = ModelSpec.load(args.fusion_run / "model.json")
    reference = json.loads(args.reference_report.read_text())
    fusion_report = json.loads((args.fusion_run / "training-report.json").read_text())
    extra_report = json.loads((args.extra_run / "training-report.json").read_text())
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if (
        manifest_sha != reference["dataset_manifest_sha256"]
        or manifest_sha != fusion_report["manifest_sha256"]
    ):
        raise ValueError("Original manifest changed")
    if (
        hashlib.sha256(args.extra_manifest.read_bytes()).hexdigest()
        != extra_report["extra_manifest_sha256"]
    ):
        raise ValueError("Extra training manifest changed")
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("FER+ weights changed")
    excluded = {Path(r["path"]).resolve() for r in reference["excluded_similarity_matches"]}
    rows = [r for r in read_manifest(args.manifest) if r["path"] not in excluded]
    extra = read_manifest(args.extra_manifest)
    if any(r["split"] != "train" for r in extra):
        raise ValueError("Extra data must be source training images")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "calibration", "test")}
    if counts != reference["split_counts"]:
        raise ValueError("Split membership changed")
    x = np.load(args.feature_run / "features.npy", mmap_mode="r")
    xf = np.load(args.fusion_run / "fer-features.npy", mmap_mode="r")
    xe = np.load(args.extra_run / "extra-features.npy", mmap_mode="r")
    total = len(x)
    if len(xe) != 2 * len(extra) or len(xf) != total:
        raise ValueError("Feature cache shape mismatch")
    rng = np.random.default_rng(plan["seed"])
    permutation = rng.permutation(len(extra))
    held = permutation[: max(1, len(extra) // 5)]
    source_train = permutation[len(held) :]
    original_train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    train = np.array(original_train + list(range(len(rows), total)))
    train = np.concatenate([train, total + source_train, total + len(extra) + source_train])
    color_val = np.array([i for i, r in enumerate(rows) if r["split"] == "calibration"])
    val = np.concatenate([color_val, total + held])

    def features(indices):
        result = np.empty((len(indices), 2304), np.float32)
        mask = indices < total
        original = indices[mask]
        result[mask] = np.concatenate([x[original], xf[original]], axis=1)
        result[~mask] = xe[indices[~mask] - total]
        return result

    domains = (train >= total).astype(np.float32)
    vd = (val >= total).astype(np.float32)
    mean = np.zeros(2304, np.float64)
    squares = np.zeros_like(mean)
    for ids in np.array_split(train, 100):
        values = features(ids).astype(np.float64)
        mean += values.sum(0)
        squares += (values * values).sum(0)
    mean = (mean / len(train)).astype(np.float32)
    std = np.maximum(
        np.sqrt(np.maximum(squares / len(train) - mean.astype(np.float64) ** 2, 0)), 0.1
    ).astype(np.float32)
    zv = (features(val) - mean) / std
    w, b = np.zeros(2304, np.float32), np.zeros(1, np.float32)
    moments = [np.zeros_like(w), np.zeros_like(b)]
    variances = [np.zeros_like(w), np.zeros_like(b)]
    best_loss, best_epoch, best, stale, step = float("inf"), 0, None, 0, 0
    history = []
    for epoch in range(1, 101):
        for positions in np.array_split(
            rng.permutation(len(train)), int(np.ceil(len(train) / 256))
        ):
            ids = train[positions]
            xb = (features(ids) - mean) / std
            target = domains[positions]
            weights = np.where(
                target == 1,
                len(train) / (2 * domains.sum()),
                len(train) / (2 * (len(train) - domains.sum())),
            )
            error = (sigmoid(xb @ w + b) - target) * weights
            gradients = [xb.T @ error / len(ids) + 0.001 * w, np.array([error.mean()], np.float32)]
            step += 1
            for parameter, g, m, v in zip([w, b], gradients, moments, variances):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                parameter -= (
                    0.0005 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
        p = np.clip(sigmoid(zv @ w + b), 1e-7, 1 - 1e-7)
        losses = -(vd * np.log(p) + (1 - vd) * np.log1p(-p))
        loss = float((losses[vd == 0].mean() + losses[vd == 1].mean()) / 2)
        history.append({"epoch": epoch, "balanced_domain_validation_nll": loss})
        print(f"Domain epoch {epoch}: balanced validation NLL {loss:.4f}", flush=True)
        if loss < best_loss - 0.0001:
            best_loss, best_epoch, best, stale = loss, epoch, (w.copy(), b.copy()), 0
        else:
            stale += 1
        if stale >= 10:
            break
    w, b = best
    fw = (w / std).astype(np.float32)
    fb = (b - mean @ fw).astype(np.float32)
    model = onnx.load(spec.path)
    params = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    fer = onnx.load(MODEL_PATH)
    fer_params = {i.name: numpy_helper.to_array(i) for i in fer.graph.initializer}
    order = [RAW_LABELS.index(name) for name in spec.labels]
    nw = fer_params["Parameter1693"][:, order].T.copy()
    nb = fer_params["Parameter1694"][order].copy()
    fv = features(val)
    color = softmax(fv @ params["fusion_head_weight"].T + params["fusion_head_bias"])
    gray = softmax(fv[:, 1280:] @ nw.T + nb)
    alpha = sigmoid(fv @ fw + fb)[:, None]
    truth = np.array(
        [spec.labels.index(rows[i]["label"]) for i in color_val]
        + [spec.labels.index(extra[i]["label"]) for i in held]
    )
    ncolor = len(color_val)
    options = []
    for strength in plan["blend_strengths"]:
        probability = color * (1 - strength * alpha) + gray * (strength * alpha)
        losses = -np.log(np.maximum(probability[np.arange(len(val)), truth], 1e-12))
        accuracy = float((probability[:ncolor].argmax(1) == truth[:ncolor]).mean())
        options.append(
            {
                "strength": strength,
                "color_nll": float(losses[:ncolor].mean()),
                "color_top1": accuracy,
                "fer_training_holdout_nll": float(losses[ncolor:].mean()),
                "balanced_nll": float((losses[:ncolor].mean() + losses[ncolor:].mean()) / 2),
            }
        )
    baseline = options[0]
    eligible = [
        r
        for r in options
        if r["color_nll"] <= baseline["color_nll"] + 0.005
        and r["color_top1"] >= baseline["color_top1"] - 0.001
    ]
    selected = min(eligible, key=lambda r: r["balanced_nll"])
    strength = selected["strength"]
    output = model.graph.node[-1].output[0]
    model.graph.node[-1].output[0] = "domain_color_logits"
    fer_feature = next(
        n.input[1]
        for n in model.graph.node
        if n.op_type == "Concat" and n.output[0] == "fusion_features"
    )
    values = [
        ("domain_native_w", nw),
        ("domain_native_b", nb),
        ("domain_gate_w", fw[None]),
        ("domain_gate_b", fb),
        ("domain_one", np.array([1.0], np.float32)),
        ("domain_strength", np.array([strength], np.float32)),
    ]
    for name, value in values:
        model.graph.initializer.append(numpy_helper.from_array(value, name))
    model.graph.node.extend(
        [
            helper.make_node("Softmax", ["domain_color_logits"], ["domain_color_prob"], axis=1),
            helper.make_node(
                "Gemm",
                [fer_feature, "domain_native_w", "domain_native_b"],
                ["domain_gray_logits"],
                transB=1,
            ),
            helper.make_node("Softmax", ["domain_gray_logits"], ["domain_gray_prob"], axis=1),
            helper.make_node(
                "Gemm",
                ["fusion_features", "domain_gate_w", "domain_gate_b"],
                ["domain_logit"],
                transB=1,
            ),
            helper.make_node("Sigmoid", ["domain_logit"], ["domain_probability"]),
            helper.make_node("Mul", ["domain_probability", "domain_strength"], ["domain_alpha"]),
            helper.make_node("Sub", ["domain_one", "domain_alpha"], ["domain_color_weight"]),
            helper.make_node("Mul", ["domain_alpha", "domain_gray_prob"], ["domain_gray_part"]),
            helper.make_node(
                "Mul", ["domain_color_weight", "domain_color_prob"], ["domain_color_part"]
            ),
            helper.make_node("Add", ["domain_gray_part", "domain_color_part"], [output]),
        ]
    )
    onnx.checker.check_model(model)
    path = args.output / "expression-domain.onnx"
    onnx.save(model, path)
    manifest = dataclasses.asdict(spec)
    manifest.update(
        path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), probabilities=True
    )
    manifest_path = args.output / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    engine = InferenceBackend(ModelSpec.load(manifest_path))
    combined = color * (1 - strength * alpha) + gray * (strength * alpha)
    canonical = [spec.labels.index(name) for name in RAW_LABELS]
    for position in (0, ncolor - 1, ncolor, len(val) - 1):
        image_path = (
            rows[int(val[position])]["path"]
            if position < ncolor
            else extra[int(held[position - ncolor])]["path"]
        )
        np.testing.assert_allclose(
            engine.predict(cv2.imread(str(image_path))), combined[position, canonical], atol=1e-5
        )
    test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    ft = features(test)
    color = softmax(ft @ params["fusion_head_weight"].T + params["fusion_head_bias"])
    gray = softmax(ft[:, 1280:] @ nw.T + nb)
    alpha = sigmoid(ft @ fw + fb)[:, None] * strength
    probabilities = color * (1 - alpha) + gray * alpha
    labels = np.array([RAW_LABELS.index(rows[i]["label"]) for i in test])
    score = metrics(labels, probabilities[:, canonical], threshold=0.5, margin=0.08)
    old = fusion_report["baseline_test"]
    gate = fusion_report["promotion_gate"]
    accepted = (
        selected["color_top1"]
        >= fusion_report["baseline_validation"]["top1"] + gate["validation_top1_minimum_gain"]
        and score["raw_top1_accuracy"] >= old["raw_top1_accuracy"]
        and score["macro_f1_present_classes"] >= old["macro_f1_present_classes"]
        and score["nll"] <= old["nll"] + gate["test_nll_maximum_increase"]
    )
    report = {
        "method": "Frozen expression experts with trained source-domain selector",
        "plan": plan,
        "domain_epochs_completed": len(history),
        "domain_selected_epoch": best_epoch,
        "domain_training_examples": len(train),
        "fer_training_holdout_images": len(held),
        "blend_validation": options,
        "selected_blend": selected,
        "primary_test": score,
        "accepted": accepted,
        "production_model_changed": False,
        "model_sha256": manifest["sha256"],
        "original_manifest_sha256": manifest_sha,
        "extra_manifest_sha256": extra_report["extra_manifest_sha256"],
        "limitations": [
            "Validation/test subsets have been reused; pretraining overlap and subject separation unknown",
            "FER source holdout comes from source training, not the external test",
            "Domain labels denote dataset origin, not a person attribute",
        ],
        "history": history,
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
