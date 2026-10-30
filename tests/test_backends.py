import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backends import AuxiliaryBackend, InferenceBackend, ModelSpec
from tests.helpers import constant_onnx


class BackendTests(unittest.TestCase):
    def test_openvino_contract_checks_paired_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            path.write_bytes(b"contract fixture, not a runnable model")
            weights = path.with_suffix(".bin")
            weights.write_bytes(b"original weights")
            spec = ModelSpec(path, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "weights_sha256"):
                spec.validate()
            spec.weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
            spec.validate()
            weights.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "weights_sha256"):
                spec.validate()

    def test_real_onnx_contract_maps_declared_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            digest = constant_onnx(path, {"scores": [0, 9]})
            spec = ModelSpec(path, digest, labels=("sad", "happy"))
            scores = InferenceBackend(spec).predict(np.zeros((80, 80), np.uint8))
            self.assertEqual(int(scores.argmax()), 1)
            self.assertEqual(float(scores[2]), 0)
            self.assertAlmostEqual(float(scores.sum()), 1)

    def test_model_checksum_and_normalization_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            digest = constant_onnx(path, {"scores": [0] * 8})
            with self.assertRaisesRegex(ValueError, "checksum"):
                ModelSpec(path, "0" * 64).validate()
            with self.assertRaises(ValueError):
                ModelSpec(path, digest, std=(0,)).validate()

    def test_auxiliary_outputs_come_from_supplied_model_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "aux.onnx"
            digest = constant_onnx(path, {"va": [-0.2, 0.7], "au": [0.1, 2.5, 4.0]})
            manifest = root / "aux.json"
            manifest.write_text(
                json.dumps(
                    {
                        "path": path.name,
                        "sha256": digest,
                        "auxiliary_outputs": [
                            {
                                "name": "va",
                                "kind": "valence_arousal",
                                "labels": ["valence", "arousal"],
                                "range": [-1, 1],
                            },
                            {
                                "name": "au",
                                "kind": "action_units",
                                "labels": ["AU01", "AU12", "AU45"],
                                "range": [0, 5],
                            },
                        ],
                    }
                )
            )
            result = AuxiliaryBackend(manifest).predict(np.zeros((80, 80), np.uint8))
            self.assertAlmostEqual(result["valence_arousal"]["valence"], -0.2)
            self.assertEqual(result["action_units"]["AU12"], 2.5)

    def test_auxiliary_range_violation_fails_instead_of_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.onnx"
            digest = constant_onnx(path, {"va": [2, 0.5]})
            manifest = root / "aux.json"
            manifest.write_text(
                json.dumps(
                    {
                        "path": path.name,
                        "sha256": digest,
                        "auxiliary_outputs": [
                            {
                                "name": "va",
                                "kind": "valence_arousal",
                                "labels": ["valence", "arousal"],
                                "range": [-1, 1],
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "range"):
                AuxiliaryBackend(manifest).predict(np.zeros((80, 80), np.uint8))
