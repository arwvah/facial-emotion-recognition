import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from expression_lab.object_detector import COCO_LABELS, ObjectDetector
from scripts.train_object_heads import quality_focal_gradient


class TrainingMathTests(unittest.TestCase):
    def test_custom_object_model_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "candidate.onnx"
            model.write_bytes(b"fixture bytes; ORT mocked")
            manifest = root / "model.json"
            data = {
                "path": model.name,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "labels": list(COCO_LABELS),
            }
            manifest.write_text(json.dumps(data))
            engine = MagicMock()
            engine.get_inputs.return_value = [
                SimpleNamespace(name="input", shape=[1, 3, 416, 416], type="tensor(float)")
            ]
            with patch(
                "expression_lab.object_detector.ort.InferenceSession", return_value=engine
            ) as constructor:
                detector = ObjectDetector(0.4, manifest)
                self.assertEqual(detector.input_name, "input")
                self.assertEqual(constructor.call_args.args[0], str(model.resolve()))
                model.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    ObjectDetector(0.4, manifest)
                data["labels"].reverse()
                manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "canonical"):
                    ObjectDetector(0.4, manifest)

    def test_quality_focal_gradient_matches_numeric_derivative(self):
        logits = np.array([-5.0, -1.0, 0.3, 2.0, 5.0], dtype=np.float64)
        target = np.array([0.0, 0.6, 0.7, 0.0, 0.9])

        def loss(z):
            p = 1 / (1 + np.exp(-z))
            bce = -target * np.log(p) - (1 - target) * np.log1p(-p)
            return (p - target) ** 2 * bce

        epsilon = 1e-5
        numeric = (loss(logits + epsilon) - loss(logits - epsilon)) / (2 * epsilon)
        p = 1 / (1 + np.exp(-logits))
        np.testing.assert_allclose(quality_focal_gradient(p, target), numeric, rtol=1e-5, atol=1e-7)

    def test_quality_focal_gradient_vanishes_at_target(self):
        p = np.array([0.2, 0.5, 0.9])
        np.testing.assert_allclose(quality_focal_gradient(p, p), 0)
