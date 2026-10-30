"""Explicit model/preprocessing contracts for interchangeable local inference engines."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .labels import RAW_LABELS


def scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities > 0
    if not positive.any() or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid probability vector.")
    scores = np.zeros_like(probabilities)
    logits = np.log(probabilities[positive]) / temperature
    scores[positive] = np.exp(logits - logits.max())
    return (scores / scores.sum()).astype(np.float32)


@dataclass
class ModelSpec:
    path: Path
    sha256: str
    labels: tuple[str, ...] = RAW_LABELS
    width: int = 64
    height: int = 64
    color: str = "gray"
    scale: float = 1.0
    mean: tuple[float, ...] = (0.0,)
    std: tuple[float, ...] = (1.0,)
    output: str | None = None
    probabilities: bool = False
    weights_sha256: str | None = None

    @classmethod
    def load(cls, manifest: Path):
        data = json.loads(manifest.read_text())
        allowed = set(cls.__dataclass_fields__)
        if set(data) - allowed:
            raise ValueError(f"Unknown model fields: {sorted(set(data) - allowed)}")
        data["path"] = (manifest.parent / data["path"]).resolve()
        for key in ("labels", "mean", "std"):
            if key in data:
                data[key] = tuple(data[key])
        spec = cls(**data)
        spec.validate()
        return spec

    def validate(self):
        if (
            self.color not in ("gray", "rgb", "bgr")
            or not 8 <= self.width <= 1024
            or not 8 <= self.height <= 1024
        ):
            raise ValueError("Invalid model input dimensions/color.")
        channels = 1 if self.color == "gray" else 3
        if len(self.mean) not in (1, channels) or len(self.std) not in (1, channels):
            raise ValueError("Mean/std must match the model's channels.")
        if not np.isfinite([self.scale, *self.mean, *self.std]).all() or any(
            x <= 0 for x in self.std
        ):
            raise ValueError("Invalid model normalization.")
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256):
            raise ValueError("Every model requires an explicit SHA-256.")
        if (
            not self.labels
            or len(set(self.labels)) != len(self.labels)
            or not set(self.labels) <= set(RAW_LABELS)
        ):
            raise ValueError("Model labels must be a unique subset of the eight canonical labels.")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError(f"Model checksum mismatch: {self.path}")
        if self.path.suffix == ".xml" and (
            not self.weights_sha256
            or hashlib.sha256(self.path.with_suffix(".bin").read_bytes()).hexdigest()
            != self.weights_sha256
        ):
            raise ValueError("OpenVINO IR requires a matching weights_sha256 for its .bin file")

    def tensor(self, crop: np.ndarray) -> np.ndarray:
        if self.color == "gray":
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        elif crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if self.color == "rgb":
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = image[:, :, None]
        normalized = (
            image.astype(np.float32) * self.scale - np.asarray(self.mean, np.float32)
        ) / np.asarray(self.std, np.float32)
        return np.ascontiguousarray(normalized.astype(np.float32).transpose(2, 0, 1)[None])

    def signature(self) -> dict:
        return {
            "sha256": self.sha256,
            "labels": list(self.labels),
            "width": self.width,
            "height": self.height,
            "color": self.color,
            "scale": self.scale,
            "mean": list(self.mean),
            "std": list(self.std),
            "output": self.output,
            "probabilities": self.probabilities,
            "weights_sha256": self.weights_sha256,
        }


class InferenceBackend:
    def __init__(self, spec: ModelSpec, engine="onnx", provider="CPUExecutionProvider"):
        spec.validate()
        self.spec = spec
        self.engine = engine
        if engine == "onnx":
            if provider not in ort.get_available_providers():
                raise ValueError(
                    f"Provider {provider} is unavailable; installed: {ort.get_available_providers()}"
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.session = ort.InferenceSession(
                str(spec.path), sess_options=options, providers=[provider]
            )
            self.input_name = self.session.get_inputs()[0].name
            inputs = self.session.get_inputs()
            expected = [1, 1 if spec.color == "gray" else 3, spec.height, spec.width]
            if (
                len(inputs) != 1
                or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 4
                or any(
                    isinstance(actual, int) and actual != wanted
                    for actual, wanted in zip(inputs[0].shape, expected)
                )
            ):
                raise ValueError("Model must expose one float32 NCHW input matching its manifest")
            if provider != "CPUExecutionProvider" and self.session.get_providers()[0] != provider:
                raise RuntimeError(f"Requested provider {provider} failed to initialize")
        elif engine == "opencv":
            self.net = cv2.dnn.readNetFromONNX(str(spec.path))
        elif engine == "openvino":
            try:
                import openvino as ov
            except ImportError as exc:
                raise RuntimeError(
                    "Install the optimize extra for OpenVINO: pip install '.[optimize]'"
                ) from exc
            self.compiled = ov.Core().compile_model(str(spec.path), "CPU")
        else:
            raise ValueError("Engine must be onnx, opencv, or openvino.")

    def run(self, tensor: np.ndarray) -> np.ndarray:
        if self.engine == "onnx":
            return self.session.run(
                [self.spec.output] if self.spec.output else None, {self.input_name: tensor}
            )[0]
        if self.engine == "opencv":
            self.net.setInput(tensor)
            return self.net.forward(self.spec.output) if self.spec.output else self.net.forward()
        return np.asarray(self.compiled([tensor])[self.compiled.output(self.spec.output or 0)])

    def predict(self, crop: np.ndarray, temperature=1.0) -> np.ndarray:
        scores = np.asarray(self.run(self.spec.tensor(crop)), np.float64).reshape(-1)
        if len(scores) != len(self.spec.labels) or not np.isfinite(scores).all():
            raise ValueError("Model output does not match its declared labels.")
        if self.spec.probabilities:
            if (scores < 0).any() or not np.isclose(scores.sum(), 1, atol=1e-3):
                raise ValueError("Model declared probabilities but returned invalid scores.")
            scores = np.log(np.clip(scores, 1e-12, 1))
        scores = scores / temperature
        scores = np.exp(scores - scores.max())
        scores /= scores.sum()
        canonical = np.zeros(8, dtype=np.float32)
        for name, value in zip(self.spec.labels, scores):
            canonical[RAW_LABELS.index(name)] = value
        return canonical


class AuxiliaryBackend:
    """Run a supplied, trained ONNX model for VA and/or action units; never infer them from labels."""

    def __init__(self, manifest: Path, provider="CPUExecutionProvider"):
        data = json.loads(manifest.read_text())
        self.outputs = data.pop("auxiliary_outputs")
        data["path"] = (manifest.parent / data["path"]).resolve()
        data["labels"] = tuple(data.get("labels", RAW_LABELS))
        self.spec = ModelSpec(**data)
        self.backend = InferenceBackend(self.spec, provider=provider)
        self.names = {item["name"] for item in self.outputs}
        if not self.names <= {x.name for x in self.backend.session.get_outputs()}:
            raise ValueError("Auxiliary model output names do not match the manifest.")
        for output in self.outputs:
            if output["kind"] not in ("valence_arousal", "action_units"):
                raise ValueError("Unknown auxiliary output kind.")
            labels = output["labels"]
            if len(set(labels)) != len(labels) or not labels:
                raise ValueError("Auxiliary labels must be unique and nonempty")
            if output["kind"] == "valence_arousal" and set(labels) != {"valence", "arousal"}:
                raise ValueError("VA outputs must explicitly label valence and arousal")
            if output["kind"] == "action_units" and not all(
                re.fullmatch(r"AU\d{1,2}", label) for label in labels
            ):
                raise ValueError("Action unit labels must be FACS AU numbers, e.g. AU01")
            if output.get("activation") not in (None, "sigmoid"):
                raise ValueError("Auxiliary activation must be omitted or sigmoid")
            lo, hi = output["range"]
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("Invalid auxiliary output range")

    def predict(self, crop: np.ndarray) -> dict:
        results: dict[str, dict[str, float] | None] = {
            "valence_arousal": None,
            "action_units": None,
        }
        tensor = self.spec.tensor(crop)
        for output in self.outputs:
            values = self.backend.session.run([output["name"]], {self.backend.input_name: tensor})[
                0
            ]
            values = np.asarray(values).reshape(-1)
            labels = output["labels"]
            if len(values) != len(labels) or not np.isfinite(values).all():
                raise ValueError("Invalid auxiliary output shape/values.")
            if output.get("activation") == "sigmoid":
                values = 1 / (1 + np.exp(-np.clip(values, -80, 80)))
            lo, hi = output["range"]
            if ((values < lo - 1e-4) | (values > hi + 1e-4)).any():
                raise ValueError("Auxiliary predictions exceed the declared range.")
            results[output["kind"]] = {name: float(value) for name, value in zip(labels, values)}
        return results
