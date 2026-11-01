"""Fit a local age regression head on frozen face-CNN features."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from expression_lab.recognition_models import hsemotion_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    records = json.loads(args.manifest.read_text())
    rng = np.random.default_rng(42)
    valid_indices = rng.permutation(
        [i for i, r in enumerate(records) if r["source_split"] == "valid"]
    )
    test_indices = set(valid_indices[len(valid_indices) // 2 :])
    seen = {}
    rows = []
    for i, r in enumerate(records):
        path = Path(r["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise ValueError(f"Image checksum mismatch: {path}")
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen[digest] = True
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Invalid image: {path}")
        age = float(r["age"])
        if not 0 <= age <= 116:
            raise ValueError("Age must be in [0,116]")
        split = "train" if r["source_split"] == "train" else "test" if i in test_indices else "val"
        rows.append((path, age, split))
    spec = hsemotion_spec()
    spec.validate()
    model = onnx.load(spec.path)
    head = model.graph.node[-1]
    if head.op_type != "Gemm":
        raise ValueError("Expected HSEmotion classifier head")
    weights = next(
        numpy_helper.to_array(i) for i in model.graph.initializer if i.name == head.input[1]
    )
    dims = weights.shape[1]
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(head.input[0], onnx.TensorProto.FLOAT, [1, dims])
    )
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    engine = ort.InferenceSession(
        model.SerializeToString(), sess_options=opt, providers=["CPUExecutionProvider"]
    )
    features = []
    for i, (path, age, split) in enumerate(rows):
        image = cv2.imread(str(path))
        features.append(
            engine.run([head.input[0]], {engine.get_inputs()[0].name: spec.tensor(image)})[0][0]
        )
        if (i + 1) % 250 == 0:
            print(f"Age features {i + 1}/{len(rows)}", flush=True)
    x = np.asarray(features, dtype=np.float64)
    y = np.array([r[1] for r in rows])
    splits = np.array([r[2] for r in rows])
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    if min(train.sum(), val.sum(), test.sum()) < 30:
        raise ValueError("Insufficient distinct samples")
    mean = x[train].mean(0)
    std = np.maximum(x[train].std(0), 0.1)
    z = (x - mean) / std
    center = y[train].mean()
    a = z[train].T @ z[train]
    b = z[train].T @ (y[train] - center)
    best = None
    for regularization in [100.0, 1000.0, 10000.0, 100000.0]:
        coef = np.linalg.solve(a + regularization * np.eye(dims), b)
        pred = np.clip(z[val] @ coef + center, 0, 116)
        mae = float(np.abs(pred - y[val]).mean())
        print("Validation MAE:", regularization, round(mae, 2), flush=True)
        if best is None or mae < best[0]:
            best = (mae, regularization, coef)
    assert best is not None
    coef = best[2]
    weight = (coef / std).astype(np.float32)
    bias = np.array([center - mean @ weight], np.float32)
    candidate = onnx.load(spec.path)
    for item in candidate.graph.initializer:
        if item.name == head.input[1]:
            item.CopyFrom(numpy_helper.from_array(weight[None], item.name))
        if item.name == head.input[2]:
            item.CopyFrom(numpy_helper.from_array(bias, item.name))
    dim = candidate.graph.output[0].type.tensor_type.shape.dim
    dim[-1].dim_value = 1
    onnx.checker.check_model(candidate)
    path = args.output / "age-candidate.onnx"
    onnx.save(candidate, path)
    data = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "model.json").write_text(json.dumps(data, indent=2))
    from expression_lab.age import AgeEstimator

    backend = AgeEstimator(args.output / "model.json")
    first = int(np.where(test)[0][0])
    result = backend.predict(cv2.imread(str(rows[first][0])))
    expected = float(np.clip(x[first] @ weight + bias[0], 0, 116))
    assert abs(result - expected) < 1e-3
    predicted = np.clip(x[test] @ weight + bias[0], 0, 116)
    mae = float(np.abs(predicted - y[test]).mean())
    baseline = float(np.abs(center - y[test]).mean())
    report = {
        "source": "https://huggingface.co/datasets/deedax/UTK-Face-Revised",
        "method": "Frozen HSEmotion features; ridge age-regression head",
        "age_range": "0-116",
        "train": int(train.sum()),
        "validation": int(val.sum()),
        "test": int(test.sum()),
        "validation_mae_years": best[0],
        "test_mae_years": mae,
        "constant_mean_baseline_mae_years": baseline,
        "selected_regularization": best[1],
        "duplicates_removed": len(records) - len(rows),
        "identity_separation": "Unknown; no reliable person IDs",
        "limitations": [
            "Source labels are approximate and dataset is for non-commercial research.",
            "This is a small subset; age estimates are not proof of age.",
            "Individual uncertainty and demographic performance are not established.",
        ],
    }
    (args.output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
