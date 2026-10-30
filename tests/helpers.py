"""Synthetic ONNX contract fixtures; never represent trained emotion/AU/VA models."""

import hashlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def constant_onnx(path, outputs):
    nodes = []
    output_info = []
    for name, values in outputs.items():
        array = np.asarray(values, dtype=np.float32)[None]
        nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array)))
        output_info.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(array.shape))
        )
    graph = helper.make_graph(
        nodes,
        "TEST_ONLY_CONSTANT_FIXTURE",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 64, 64])],
        output_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.save(model, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()
