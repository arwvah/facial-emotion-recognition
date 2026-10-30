"""Build explicit OpenVINO/INT8 candidates; never replace the production model automatically."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from .backends import ModelSpec
from .emotion import MODEL_PATH, MODEL_SHA256
from .evaluation import benchmark, read_manifest, write_json


def quantize_candidate(spec, calibration_rows, output):
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self, input_name):
            self.input_name = input_name
            self.items = iter(calibration_rows)

        def get_next(self):
            row = next(self.items, None)
            if row is None:
                return None
            image = cv2.imread(str(row["path"]))
            if image is None:
                raise ValueError("Cannot decode quantization calibration image")
            return {self.input_name: spec.tensor(image)}

    spec.validate()
    model = onnx.load(str(spec.path))
    if any(item.version < 13 for item in model.opset_import if item.domain in ("", "ai.onnx")):
        model = onnx.version_converter.convert_version(model, 13)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        converted = Path(directory) / "converted.onnx"
        onnx.save(model, str(converted))
        quantize_static(
            str(converted),
            str(output),
            Reader(model.graph.input[0].name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        )
    candidate = replace(
        spec, path=output.resolve(), sha256=hashlib.sha256(output.read_bytes()).hexdigest()
    )
    manifest = output.with_suffix(".json")
    data = asdict(candidate)
    data["path"] = output.name
    write_json(manifest, data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("quantize", "openvino"))
    parser.add_argument("--model-spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, help="CSV with explicit calibration split (cropped faces)"
    )
    parser.add_argument("--evaluation", type=Path, help="CSV containing the independent test split")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        related = [
            args.output,
            args.output.with_suffix(".json"),
            args.output.with_suffix(".validation.json"),
        ]
        if any(path.exists() for path in related):
            raise ValueError("Candidate output/manifest exists; choose a new path")
        if not 0 <= args.max_accuracy_drop <= 1:
            raise ValueError("Accuracy tolerance must be in [0,1]")
        spec = (
            ModelSpec.load(args.model_spec)
            if args.model_spec
            else ModelSpec(MODEL_PATH, MODEL_SHA256)
        )
        spec.validate()
        if args.command == "openvino":
            import openvino as ov

            if args.output.suffix != ".xml" or args.output.with_suffix(".bin").exists():
                raise ValueError("Use a new .xml path (paired .bin must not already exist)")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            ov.save_model(
                ov.Core().read_model(str(spec.path)), str(args.output), compress_to_fp16=False
            )
            exported = replace(
                spec,
                path=args.output.resolve(),
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                weights_sha256=hashlib.sha256(
                    args.output.with_suffix(".bin").read_bytes()
                ).hexdigest(),
            )
            manifest_data = asdict(exported)
            manifest_data["path"] = args.output.name
            write_json(args.output.with_suffix(".json"), manifest_data)
            print(
                f"Exported OpenVINO IR: {args.output}; validate with the same held-out dataset before use"
            )
            return 0
        if args.output.suffix != ".onnx" or not args.calibration:
            raise ValueError("quantize requires --calibration CSV and a new .onnx output")
        all_rows = read_manifest(args.calibration, require_labels=False)
        rows = [row for row in all_rows if row["split"] == "calibration"]
        if not rows:
            raise ValueError("No calibration-split samples")
        evaluation_rows = read_manifest(args.evaluation) if args.evaluation else []
        tests = [row for row in evaluation_rows if row["split"] == "test"]
        if args.evaluation and not tests:
            raise ValueError("Evaluation manifest has no test samples")
        calibration_paths = {row["path"] for row in rows}
        if calibration_paths & {row["path"] for row in tests}:
            raise ValueError("Quantization calibration and evaluation sets overlap")
        # Also guard identical image copies and supplied subject IDs across the two manifests.
        hashes = {hashlib.sha256(row["path"].read_bytes()).hexdigest() for row in rows}
        subjects = {row.get("subject_id") for row in rows if row.get("subject_id")}
        if any(
            hashlib.sha256(row["path"].read_bytes()).hexdigest() in hashes
            or row.get("subject_id") in subjects
            for row in tests
        ):
            raise ValueError("Calibration/test images or subjects overlap")
        manifest = quantize_candidate(spec, rows, args.output)
        validation = {
            "accepted": None,
            "status": "requires_held_out_evaluation",
            "candidate_manifest": str(manifest),
            "default_model_changed": False,
        }
        if tests:
            base_config = {"align": False}
            if args.model_spec:
                base_config["model_spec"] = args.model_spec
            baseline = benchmark(tests, base_config, threshold=0)
            candidate = benchmark(tests, {"model_spec": manifest, "align": False}, threshold=0)
            drop = baseline["raw_top1_accuracy"] - candidate["raw_top1_accuracy"]
            validation.update(
                {
                    "accepted": drop <= args.max_accuracy_drop,
                    "status": "evaluated",
                    "accuracy_drop": drop,
                    "max_accuracy_drop": args.max_accuracy_drop,
                    "baseline": baseline,
                    "candidate": candidate,
                }
            )
        write_json(args.output.with_suffix(".validation.json"), validation)
        print(json.dumps(validation, indent=2))
        return 0 if validation["accepted"] is not False else 1
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
