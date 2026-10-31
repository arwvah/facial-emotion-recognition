import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from expression_lab.app import parse_args, run
from expression_lab.emotion import RAW_LABELS, FaceResult, classify
from expression_lab.evaluation import metrics
from expression_lab.object_detector import ObjectDetector, decode_outputs, letterbox, merge_objects
from expression_lab.recognition_models import hsemotion_spec
from expression_lab.service import create_app
from expression_lab.tracking import ExpressionTracker
from expression_lab.vision import VisionWorker
from tests.test_service import FakeRecognizer


def outputs():
    classes, boxes = [], []
    for stride in (8, 16, 32):
        count = (416 // stride) ** 2
        scores = np.zeros((1, count, 80), np.float32)
        distances = np.full((1, count, 32), -100, np.float32)
        distances[:, :, [1, 9, 17, 25]] = 100  # one stride in each direction
        classes.append(scores)
        boxes.append(distances)
    return classes + boxes


class RecognitionTests(unittest.TestCase):
    def test_evaluation_reports_ambiguity_as_abstention(self):
        probabilities = [[0.51, 0.48, 0.01, 0, 0, 0, 0, 0]]
        result = metrics([0], probabilities, margin=0.08)
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["raw_top1_accuracy"], 1)
        self.assertEqual(result["confusion_matrix"][0][8], 1)

    def test_invalid_recognition_options_fail_before_loading_models(self):
        for options in (
            ["--object-threshold", "nan"],
            ["--expression-margin", "1.1"],
            ["--face-detector-size", "100"],
            ["--object-detector", "unknown"],
            ["--setup-recognition", "--check"],
        ):
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(options)

    def test_hse_color_normalization_and_label_order(self):
        spec = hsemotion_spec()
        frame = np.zeros((20, 30, 3), np.uint8)
        frame[:, :, 2] = 255  # BGR red -> first RGB channel
        tensor = spec.tensor(frame)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        np.testing.assert_allclose(
            tensor[0, :, 0, 0], [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225], rtol=1e-6
        )
        self.assertEqual(spec.labels[4:7], ("happy", "neutral", "sad"))
        self.assertEqual(spec.signature()["interpolation"], "linear")

    def test_margin_survives_tracking_without_overriding_quality(self):
        scores = np.array([0.51, 0.48, 0.01, 0, 0, 0, 0, 0])
        self.assertEqual(classify(scores, 0.5, 0.08)[0], "uncertain")
        face = FaceResult((0, 0, 80, 80), "neutral", 0.51, dict(zip(RAW_LABELS, scores)))
        tracker = ExpressionTracker(margin=0.08)
        result = tracker.update([face], 0)[0]
        self.assertEqual(result.uncertainty_reason, "ambiguous_expression")
        face.uncertainty_reason = "image_quality"
        self.assertEqual(tracker.update([face], 0.1)[0].uncertainty_reason, "image_quality")

    def test_decode_distribution_and_letterbox_coordinates(self):
        raw = outputs()
        raw[0][0, 26 * 52 + 26, 16] = 0.9  # dog at image center
        boxes, scores = decode_outputs(raw)
        np.testing.assert_allclose(boxes[0], [203.5, 203.5, 219.5, 219.5])
        self.assertEqual(scores.argmax(axis=1)[0], 16)
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.threshold, detector.input_name = 0.4, "input"
        detector.session = MagicMock()
        detector.session.run.return_value = raw
        frame = np.full((200, 400, 3), 128, np.uint8)
        image, mapping = letterbox(frame)
        self.assertEqual(mapping, (0, 104, 1.04, 1.04))
        self.assertFalse(image[:104].any())
        detected = detector.detect(frame)
        self.assertEqual(detected[0]["label"], "dog")
        self.assertEqual(detected[0]["box_xyxy"], [196, 96, 211, 111])
        self.assertEqual(detected[0]["source"], "nanodet")

    def test_empty_or_malformed_detector_output(self):
        boxes, _ = decode_outputs(outputs())
        self.assertEqual(len(boxes), 0)
        with self.assertRaises(ValueError):
            decode_outputs([])
        raw = outputs()
        raw[0][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            decode_outputs(raw)

    def test_merge_keeps_novel_regions_and_fresh_detector_geometry(self):
        fresh = [{"label": "cup", "box_xyxy": [10, 10, 90, 90], "source": "nanodet"}]
        described = [
            {"label": "mug", "box_xyxy": [12, 12, 92, 92]},
            {"label": "circuit board", "box_xyxy": [100, 100, 150, 150]},
        ]
        result = merge_objects(fresh, described)
        self.assertEqual([x["label"] for x in result], ["cup", "circuit board"])
        self.assertEqual(result[1]["source"], "qwen")
        self.assertNotIn("source", described[1])

    def test_region_inspection_restores_original_frame_coordinates(self):
        client = MagicMock()
        client.describe.return_value = {
            "status": "ok",
            "frame_size_wh": [40, 30],
            "objects": [{"label": "chip", "box_xyxy": [0, 0, 40, 30]}],
        }
        worker = VisionWorker(client)
        try:
            frame = np.zeros((100, 120, 3), np.uint8)
            worker.submit(frame, "Identify", 8, 1.5, region=(20, 10, 40, 30))
            result = worker.poll(wait=True)
            self.assertEqual(client.describe.call_args.args[0].shape, (30, 40, 3))
            self.assertEqual(result["objects"][0]["box_xyxy"], [20, 10, 60, 40])
            self.assertEqual(result["frame_size_wh"], [120, 100])
            self.assertEqual(result["source_frame"], 8)
            with self.assertRaises(ValueError):
                worker.submit(frame, "Identify", 9, 2, region=(110, 10, 40, 30))
        finally:
            worker.close()

    def test_setup_recognition_respects_no_storage(self):
        with patch("expression_lab.recognition_models.setup_recognition") as setup:
            with self.assertRaisesRegex(ValueError, "no-storage"):
                run(parse_args(["--setup-recognition", "--no-storage"]))
            setup.assert_not_called()

    def test_detector_cli_and_api_work_without_language_model(self):
        objects = [
            {"label": "cup", "box_xyxy": [10, 10, 60, 70], "source": "nanodet", "confidence": 0.9}
        ]
        detector = MagicMock()
        detector.detect.return_value = objects
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((80, 80, 3), np.uint8)
            path = root / "frame.png"
            cv2.imwrite(str(path), frame)
            with (
                patch("expression_lab.app.EmotionRecognizer", return_value=FakeRecognizer()),
                patch("expression_lab.object_detector.ObjectDetector", return_value=detector),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                run(
                    parse_args(
                        [
                            "--image",
                            str(path),
                            "--headless",
                            "--no-storage",
                            "--object-detector",
                            "nanodet",
                        ]
                    )
                )
            record = json.loads(stdout.getvalue())
            self.assertEqual(record["object_overlays"], objects)
            self.assertNotIn("vision", record)
            config = root / "api.yaml"
            config.write_text("object_detector: nanodet\n")
            with patch.dict(os.environ, {"EXPRESSION_LAB_API_TOKEN": ""}):
                app = create_app(
                    config, recognizer_factory=FakeRecognizer, object_factory=lambda: detector
                )
            with TestClient(app) as client:
                response = client.post("/analyze", content=path.read_bytes())
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["objects"], objects)
                self.assertEqual(client.get("/capabilities").json()["object_detector"], "nanodet")
