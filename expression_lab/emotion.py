"""Face detection, verified model installation, and FER+ CNN inference."""

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2.data import haarcascades

from .labels import RAW_LABELS, TARGET_LABELS

__all__ = [
    "RAW_LABELS",
    "TARGET_LABELS",
    "EmotionRecognizer",
    "FaceResult",
    "checksum",
    "classify",
    "download_detector",
    "download_model",
    "softmax",
]

SOURCE_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(
    os.environ.get(
        "EXPRESSION_LAB_HOME",
        SOURCE_ROOT
        if (SOURCE_ROOT / "pyproject.toml").is_file()
        else Path.home() / ".cache" / "expression-lab",
    )
)
MODEL_PATH = ROOT / "models" / "emotion-ferplus-8.onnx"
MODEL_COMMIT = "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177"
MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/"
    "4d016bfebdb4122b1f37511c5a9d40b5c87054a8/emotion-ferplus-8.onnx"
)
MODEL_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
MODEL_SIZE = 35_040_571
DETECTOR_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DETECTOR_COMMIT = "f12e12798e8314f7c074a6656816c048dcc95b7a"
DETECTOR_URL = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{DETECTOR_COMMIT}/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECTOR_SIZE = 232_589


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(path: Path = MODEL_PATH) -> Path:
    """Download only on explicit setup, with a pinned upstream SHA-256 check."""
    return download_asset(path, MODEL_URL, MODEL_SHA256, MODEL_SIZE, "FER+ CNN")


def download_detector(path: Path = DETECTOR_PATH) -> Path:
    return download_asset(path, DETECTOR_URL, DETECTOR_SHA256, DETECTOR_SIZE, "YuNet detector")


def download_asset(path: Path, url: str, expected_hash: str, size: int, name: str) -> Path:
    if path.exists():
        if checksum(path) != expected_hash:
            raise ValueError(f"Model checksum mismatch: {path}. Remove it and run setup again.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        print(f"Downloading {name} ({size / 1e6:.1f} MB)...", file=sys.stderr, flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "ExpressionLab/0.2"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > size:
                    raise ValueError("Model download exceeded the expected size.")
                out.write(block)
                if received // (5 * 1024 * 1024) != (received - len(block)) // (5 * 1024 * 1024):
                    print(f"Downloaded {received / size:.0%}", file=sys.stderr, flush=True)
        if temporary.stat().st_size != size or checksum(temporary) != expected_hash:
            raise ValueError("Downloaded model failed size/checksum verification.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.shape != (8,) or not np.isfinite(logits).all():
        raise ValueError("Expected eight finite FER+ output scores.")
    scores = np.exp(logits - np.max(logits))
    return scores / scores.sum()


def classify(probabilities: np.ndarray, threshold: float) -> tuple[str, float]:
    winner = int(np.argmax(probabilities))
    confidence = float(probabilities[winner])
    label = RAW_LABELS[winner]
    if confidence < threshold:
        label = "uncertain"
    return label, confidence


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    probabilities: dict[str, float]
    track_id: int | None = None
    raw_probabilities: dict[str, float] | None = None
    detection_confidence: float | None = None
    landmarks: list | None = None
    quality: dict | None = None
    aligned: bool = False
    uncertainty_reason: str | None = None
    valence_arousal: dict | None = None
    action_units: dict | None = None
    fresh_prediction: bool = True
    classification_skipped: bool = False

    def as_dict(self) -> dict:
        return {
            "box_xywh": list(self.box),
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "raw_probabilities": self.raw_probabilities or self.probabilities,
            "track_id": self.track_id,
            "detection_confidence": self.detection_confidence,
            "landmarks": self.landmarks,
            "quality": self.quality,
            "aligned": self.aligned,
            "uncertainty_reason": self.uncertainty_reason,
            "valence_arousal": self.valence_arousal,
            "action_units": self.action_units,
            "fresh_prediction": self.fresh_prediction,
            "classification_skipped": self.classification_skipped,
        }


class EmotionRecognizer:
    def __init__(
        self,
        threshold=0.5,
        detector="yunet",
        *,
        engine="onnx",
        provider="CPUExecutionProvider",
        model_spec=None,
        temperature=1.0,
        align=True,
        tta=False,
        quality=None,
        quality_mode="skip",
        auxiliary=None,
    ):
        from .backends import AuxiliaryBackend, InferenceBackend, ModelSpec
        from .quality import QualityPolicy

        if not 0 <= threshold <= 1 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Threshold must be in [0,1] and temperature must be positive.")
        if quality_mode not in ("skip", "flag"):
            raise ValueError("Quality mode must be skip or flag.")
        spec = (
            ModelSpec.load(Path(model_spec)) if model_spec else ModelSpec(MODEL_PATH, MODEL_SHA256)
        )
        if not spec.path.exists():
            raise FileNotFoundError("CNN model missing. Run: expression-lab --setup")
        self.backend = InferenceBackend(spec, engine, provider)
        if engine == "onnx":
            self.session = self.backend.session
            self.input_name = self.backend.input_name
        self.temperature = temperature
        self.align = align
        self.tta = tta
        self.quality_policy = quality or QualityPolicy()
        self.quality_mode = quality_mode
        self.auxiliary = AuxiliaryBackend(Path(auxiliary), provider) if auxiliary else None
        self.detector_kind = detector
        self.detector: Any
        if detector == "yunet":
            if not DETECTOR_PATH.exists():
                raise FileNotFoundError("YuNet model missing. Run --setup or use --detector haar.")
            if checksum(DETECTOR_PATH) != DETECTOR_SHA256:
                raise ValueError("YuNet checksum mismatch. Remove the model and run --setup.")
            self.detector = cv2.FaceDetectorYN.create(
                str(DETECTOR_PATH), "", (320, 320), 0.85, 0.3, 5000
            )
        elif detector == "haar":
            self.detector = cv2.CascadeClassifier(
                haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.detector.empty():
                raise RuntimeError("OpenCV face detector could not be loaded.")
        else:
            raise ValueError("Detector must be yunet or haar.")
        self.threshold = threshold

    def predict_crop(self, gray_crop: np.ndarray) -> np.ndarray:
        from .backends import scale_probabilities
        from .quality import augment

        if not hasattr(self, "backend"):
            # The default FER+ tensor contract is also independently tested.
            resized = cv2.resize(gray_crop, (64, 64), interpolation=cv2.INTER_AREA)
            tensor = resized.astype(np.float32)[None, None, :, :]
            return softmax(self.session.run(None, {self.input_name: tensor})[0])
        variants = augment(gray_crop, self.tta)
        scores = np.mean([self.backend.predict(crop) for crop in variants], axis=0)
        return scale_probabilities(scores, self.temperature)

    def classify_observations(
        self, frame: np.ndarray, observations: list[FaceResult]
    ) -> list[FaceResult]:
        from dataclasses import replace

        from .quality import aligned_crop

        results = []
        for observation in observations:
            x, y, w, h = observation.box
            original = frame[y : y + h, x : x + w]
            policy = getattr(self, "quality_policy", None)
            quality = policy.measure(original) if policy else None
            crop, aligned = aligned_crop(
                frame,
                observation.box,
                observation.landmarks if getattr(self, "align", False) else None,
            )
            bad = quality is not None and not quality["accepted"]
            if bad and getattr(self, "quality_mode", "skip") == "skip":
                scores = np.full(8, 0.125, dtype=np.float32)
            else:
                # Preserve the tested grayscale FER+ input path; RGB adapters convert below.
                model = getattr(getattr(self, "backend", None), "spec", None)
                input_crop = (
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if model is None or model.color == "gray"
                    else crop
                )
                scores = self.predict_crop(input_crop)
            label, confidence = classify(scores, self.threshold)
            reason = "image_quality" if bad else "low_confidence" if label == "uncertain" else None
            if bad:
                label = "uncertain"
            auxiliary_backend = getattr(self, "auxiliary", None)
            auxiliary = (
                auxiliary_backend.predict(crop) if auxiliary_backend is not None and not bad else {}
            )
            results.append(
                replace(
                    observation,
                    label=label,
                    confidence=confidence,
                    probabilities={name: float(value) for name, value in zip(RAW_LABELS, scores)},
                    quality=quality,
                    aligned=aligned,
                    uncertainty_reason=reason,
                    valence_arousal=auxiliary.get("valence_arousal"),
                    action_units=auxiliary.get("action_units"),
                    classification_skipped=bool(
                        bad and getattr(self, "quality_mode", "skip") == "skip"
                    ),
                )
            )
        return results

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        return self.classify_observations(frame, self.locate(frame))

    def locate(self, frame: np.ndarray) -> list[FaceResult]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("Expected a nonempty BGR image.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale detection only; classify crops from the original frame.
        scale = min(1.0, 640 / max(gray.shape))
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
        detection_scores: list[float | None]
        if self.detector_kind == "yunet":
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, detections = self.detector.detect(small)
            boxes = [] if detections is None else [row[:4] for row in detections]
            detection_scores = [] if detections is None else [float(row[14]) for row in detections]
        else:
            boxes = self.detector.detectMultiScale(
                cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            detection_scores = [None] * len(boxes)
        results = []
        height, width = gray.shape
        for index, (box, detector_score) in enumerate(zip(boxes, detection_scores)):
            bx, by, bw, bh = box
            x, y = max(0, round(float(bx) * sx)), max(0, round(float(by) * sy))
            right = min(width, round(float(bx + bw) * sx))
            bottom = min(height, round(float(by + bh) * sy))
            w, h = right - x, bottom - y
            if w <= 0 or h <= 0:
                continue
            landmarks = None
            if self.detector_kind == "yunet":
                points = np.asarray(detections[index][4:14]).reshape(5, 2) * np.array([sx, sy])
                landmarks = points.tolist()
            results.append(
                FaceResult(
                    (x, y, w, h),
                    "uncertain",
                    0.0,
                    dict.fromkeys(RAW_LABELS, 0.125),
                    detection_confidence=detector_score,
                    landmarks=landmarks,
                )
            )
        return sorted(results, key=lambda result: result.box[2] * result.box[3], reverse=True)
