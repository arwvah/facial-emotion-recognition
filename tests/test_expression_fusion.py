"""Check that the extra frozen CNN sees the intended grayscale pixel values."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from expression_lab.backends import ModelSpec
from scripts import train_expression_fusion as fusion


class FusionPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fer.onnx"
        graph = helper.make_graph(
            [
                helper.make_node("Flatten", ["pixels"], ["flat"], axis=1),
                helper.make_node("Dropout", ["flat"], ["features"], ratio=0.5),
                helper.make_node("MatMul", ["features", "weight"], ["logits"]),
                helper.make_node("Add", ["logits", "bias"], ["scores"]),
            ],
            "tiny-fer",
            [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 1, 64, 64])],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8])],
            [
                numpy_helper.from_array(np.zeros((4096, 8), np.float32), "weight"),
                numpy_helper.from_array(np.zeros(8, np.float32), "bias"),
            ],
        )
        fer = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 8)])
        fer.ir_version = 6
        onnx.save(fer, self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        tensor = helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [1, 3, 224, 224])
        self.baseline = helper.make_model(
            helper.make_graph([], "input-contract", [tensor], [tensor]),
            opset_imports=[helper.make_opsetid("", 11)],
        )
        self.baseline.ir_version = 6
        self.spec = ModelSpec(
            self.path,
            self.digest,
            width=224,
            height=224,
            color="rgb",
            scale=1 / 255,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation="linear",
        )

    def test_denormalization_channel_order_and_resize_match_pixels(self):
        image = np.zeros((224, 224, 3), np.uint8)
        image[:91, :, 2] = 201  # Red: BGR source -> RGB model input.
        image[:, 57:, 1] = 43
        image[71:, :151, 0] = 99
        with patch.multiple(fusion, MODEL_PATH=self.path, MODEL_SHA256=self.digest):
            branch, _, dim = fusion.fer_branch(self.spec, self.baseline)
        self.assertEqual(dim, 4096)
        self.assertNotIn("Dropout", [node.op_type for node in branch.graph.node])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(branch.SerializeToString(), sess_options=options)
        actual = session.run(None, {"rgb": self.spec.tensor(image)})[0].reshape(64, 64)
        expected = cv2.transform(
            image.astype(np.float32), np.array([[0.114, 0.587, 0.299]], np.float32)
        )
        expected = cv2.resize(expected, (64, 64), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)

    def test_modified_second_model_is_rejected(self):
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        with (
            patch.multiple(fusion, MODEL_PATH=self.path, MODEL_SHA256=self.digest),
            self.assertRaisesRegex(ValueError, "checksum mismatch"),
        ):
            fusion.fer_branch(self.spec, self.baseline)
