# SPDX-License-Identifier: AGPL-3.0-only
"""Export a selected local checkpoint and verify the application input/output contract."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.yolo_expression.environment import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="A validation image")
    args = parser.parse_args()
    configure(args.checkpoint.parent / "settings")

    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    from expression_lab.backends import ModelSpec

    model = YOLO(str(args.checkpoint))
    path = Path(
        model.export(format="onnx", imgsz=224, batch=1, dynamic=False, simplify=False, opset=17)
    )
    spec = ModelSpec(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        labels=tuple(model.names[i] for i in range(8)),
        width=224,
        height=224,
        color="rgb",
        scale=1 / 255,
        mean=(0.0,),
        std=(1.0,),
        probabilities=True,
        interpolation="linear",
    )
    spec.validate()
    crop = cv2.imread(str(args.image))
    if crop is None:
        raise ValueError("Cannot read validation image")
    tensor = spec.tensor(crop)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    model.model.cpu().eval()
    with torch.inference_mode():
        expected = model.model(torch.from_numpy(tensor))
        if isinstance(expected, tuple):
            expected = expected[0]
        expected = expected.numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    if actual.shape != (1, 8) or not np.isclose(actual.sum(), 1, atol=1e-5):
        raise ValueError("Expected eight class probabilities")
    data = {"path": path.name, **spec.signature()}
    manifest = path.parent / "model.json"
    manifest.write_text(json.dumps(data, indent=2))
    print(
        json.dumps(
            {"manifest": str(manifest), "max_export_error": float(abs(actual - expected).max())}
        )
    )


if __name__ == "__main__":
    main()
