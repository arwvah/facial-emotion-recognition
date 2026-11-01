"""Experimental transfer learning for NanoDet classification heads, preserving box regression."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from expression_lab.object_detector import COCO_LABELS, ObjectDetector, decode_outputs, letterbox
from expression_lab.recognition_models import NANODET_PATH


def quality_focal_gradient(probability, target):
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    error = p - target
    bce = -target * np.log(p) - (1 - target) * np.log1p(-p)
    return error**3 + 2 * error * p * (1 - p) * bce


def session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def load_split(root, split):
    data = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
    categories = {c["id"]: COCO_LABELS.index(c["name"]) for c in data["categories"]}
    image_info = {image["id"]: image for image in data["images"]}
    annotations = {}
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        image = image_info[a["image_id"]]
        if (
            not np.isfinite([x, y, w, h]).all()
            or min(x, y) < 0
            or min(w, h) <= 0
            or x + w > image["width"] + 1
            or y + h > image["height"] + 1
        ):
            raise ValueError(f"Invalid bounding box in {split}: annotation {a['id']}")
        annotations.setdefault(a["image_id"], []).append(
            ([x, y, x + w, y + h], categories[a["category_id"]])
        )
    return [
        (root / split / "images" / im["file_name"], annotations.get(im["id"], []))
        for im in data["images"]
    ]


def ious(boxes, truth):
    lt = np.maximum(boxes[:, None, :2], truth[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], truth[None, :, 2:])
    intersection = np.maximum(rb - lt, 0).prod(axis=2)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    gt_area = np.maximum(truth[:, 2:] - truth[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area[:, None] + gt_area[None, :] - intersection, 1e-9)


def detector(model):
    obj = ObjectDetector.__new__(ObjectDetector)
    obj.session = session(model)
    obj.input_name = obj.session.get_inputs()[0].name
    obj.threshold = 0.05
    return obj


def evaluate(model, samples):
    obj = detector(model)
    predictions = [[] for _ in COCO_LABELS]
    totals = np.zeros(80, int)
    for path, annotations in samples:
        frame = cv2.imread(str(path))
        found = obj.detect(frame)
        for cls in range(80):
            gt = np.array([box for box, label in annotations if label == cls]).reshape(-1, 4)
            totals[cls] += len(gt)
            used = set()
            for item in sorted(
                (p for p in found if p["label"] == COCO_LABELS[cls]),
                key=lambda p: p["confidence"],
                reverse=True,
            ):
                correct = False
                if len(gt):
                    scores = ious(np.array([item["box_xyxy"]]), gt)[0]
                    choices = [j for j in np.argsort(-scores) if j not in used and scores[j] >= 0.5]
                    if choices:
                        used.add(choices[0])
                        correct = True
                predictions[cls].append((item["confidence"], correct))
    aps = {}
    for cls, name in enumerate(COCO_LABELS):
        if not totals[cls]:
            continue
        ordered = sorted(predictions[cls], reverse=True)
        tp = np.cumsum([int(x[1]) for x in ordered])
        fp = np.cumsum([int(not x[1]) for x in ordered])
        recall = tp / totals[cls]
        precision = tp / np.maximum(tp + fp, 1)
        aps[name] = float(
            np.mean(
                [
                    precision[recall >= r].max() if (recall >= r).any() else 0
                    for r in np.linspace(0, 1, 101)
                ]
            )
        )
    return {
        "mean_ap50": float(np.mean(list(aps.values()))),
        "per_class_ap50": aps,
        "metric": "101-point interpolated box AP at IoU .5; score >= .05; max 30 detections; crowd annotations ignored, not official COCOeval",
    }


def replace_heads(base, heads, weights, biases):
    model = onnx.ModelProto()
    model.CopyFrom(base)
    values = {}
    for head, w, b in zip(heads, weights, biases):
        old = {
            i.name: numpy_helper.to_array(i).copy()
            for i in base.graph.initializer
            if i.name in head.input[1:]
        }
        old[head.input[1]][:80, :, 0, 0] = w
        old[head.input[2]][:80] = b
        values.update(old)
    for item in model.graph.initializer:
        if item.name in values:
            item.CopyFrom(numpy_helper.from_array(values[item.name], item.name))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--quality-focal", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    start = time.time()
    rng = np.random.default_rng(42)
    base = onnx.load(NANODET_PATH)
    heads = [n for n in base.graph.node if n.op_type == "Conv" and "head.gfl_cls." in n.input[1]]
    assert len(heads) == 3
    initial = {i.name: numpy_helper.to_array(i).copy() for i in base.graph.initializer}
    weights = [initial[h.input[1]][:80, :, 0, 0].copy() for h in heads]
    biases = [initial[h.input[2]][:80].copy() for h in heads]
    anchors = [w.copy() for w in weights]
    extraction = onnx.ModelProto()
    extraction.CopyFrom(base)
    for h in heads:
        extraction.graph.output.append(
            onnx.helper.make_tensor_value_info(h.input[0], onnx.TensorProto.FLOAT, None)
        )
    engine = session(extraction)
    samples = {s: load_split(args.root, s) for s in ["train", "val", "test"]}
    xs = [[] for _ in heads]
    ys = [[] for _ in heads]
    for index, (path, annotations) in enumerate(samples["train"]):
        frame = cv2.imread(str(path))
        image, (left, top, sx, sy) = letterbox(frame)
        tensor = (
            ((image.astype(np.float32) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395])
            .astype(np.float32)
            .transpose(2, 0, 1)[None]
        )
        out = engine.run(None, {engine.get_inputs()[0].name: np.ascontiguousarray(tensor)})
        truth = np.array([box for box, label in annotations]).reshape(-1, 4)
        if not len(truth):
            continue
        truth = truth * [sx, sy, sx, sy] + [left, top, left, top]
        labels = np.array([label for box, label in annotations])
        offset = 0
        for level, feature in enumerate(out[6:]):
            # Decode all anchor boxes from the frozen distributional regression outputs.
            boxes, _ = decode_outputs(out[:6], threshold=0)
            n = feature.shape[2] * feature.shape[3]
            bb = boxes[offset : offset + n]
            offset += n
            overlap = ious(bb, truth)
            best = overlap.argmax(axis=1)
            quality = overlap.max(axis=1)
            pos = np.where(quality >= 0.5)[0]
            neg = np.where(quality < 0.2)[0]
            if len(pos) > 64:
                pos = rng.choice(pos, 64, replace=False)
            if len(neg) > 64:
                neg = rng.choice(neg, 64, replace=False)
            chosen = np.concatenate([pos, neg])
            targets = np.zeros((len(chosen), 80), np.float32)
            targets[np.arange(len(pos)), labels[best[pos]]] = (
                quality[pos] if args.quality_focal else 1
            )
            xs[level].append(feature[0].reshape(96, -1).T[chosen])
            ys[level].append(targets)
        if (index + 1) % 50 == 0:
            print(f"Object features {index + 1}/{len(samples['train'])}", flush=True)
    xs = [np.concatenate(v) for v in xs]
    ys = [np.concatenate(v) for v in ys]
    baseline_val = evaluate(base, samples["val"])
    best_ap = baseline_val["mean_ap50"]
    best = base
    selected = 0
    history = []
    stale = 0
    print("Baseline object validation AP50", best_ap, flush=True)
    for epoch in range(1, args.epochs + 1):
        for level in range(3):
            for batch in np.array_split(
                rng.permutation(len(xs[level])), max(1, int(np.ceil(len(xs[level]) / 256)))
            ):
                x, y = xs[level][batch], ys[level][batch]
                p = 1 / (1 + np.exp(-np.clip(x @ weights[level].T + biases[level], -30, 30)))
                error = (
                    quality_focal_gradient(p, y)
                    if args.quality_focal
                    else (p - y) * np.where(y > 0, 20, 1)
                )
                weights[level] -= 0.0001 * (
                    error.T @ x / len(batch) + 0.05 * (weights[level] - anchors[level])
                )
                biases[level] -= 0.0001 * error.mean(axis=0)
        current = replace_heads(base, heads, weights, biases)
        if epoch % 2 == 0 or epoch == args.epochs:
            score = evaluate(current, samples["val"])["mean_ap50"]
            history.append({"epoch": epoch, "validation_ap50": score})
            print(f"Object epoch {epoch}: validation AP50 {score:.4f}", flush=True)
            if score > best_ap + 0.0001:
                best_ap, best, selected = score, current, epoch
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                print("Object validation plateau; stopping", flush=True)
                break
    onnx.checker.check_model(current)
    onnx.save(current, args.output / "last-trained.onnx")
    onnx.save(best, args.output / "selected-model.onnx")
    report = {
        "method": "Frozen NanoDet backbone and regression; SGD classification heads only",
        "epochs_completed": epoch,
        "maximum_epochs": args.epochs,
        "classification_loss": "quality focal beta=2" if args.quality_focal else "weighted BCE",
        "selected_epoch": selected,
        "seed": 42,
        "baseline_validation": baseline_val,
        "history": history,
        "baseline_test": None if args.validation_only else evaluate(base, samples["test"]),
        "selected_test": None if args.validation_only else evaluate(best, samples["test"]),
        "last_trained_test": None if args.validation_only else evaluate(current, samples["test"]),
        "selected_sha256": hashlib.sha256(
            (args.output / "selected-model.onnx").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.time() - start,
        "production_model_changed": False,
        "limitations": [
            "Small COCO validation-derived subset; pretrained overlap unknown.",
            "Crowd handling and AP implementation differ from official COCOeval.",
            "Frozen regression limits localization learning.",
            "Unselected trained candidates must not be deployed as improvements.",
        ],
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_epoch": selected,
                "baseline_test_ap50": None
                if args.validation_only
                else report["baseline_test"]["mean_ap50"],
                "selected_test_ap50": None
                if args.validation_only
                else report["selected_test"]["mean_ap50"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
