import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import dashboard, parse_args
from emotion import TARGET_LABELS, EmotionRecognizer, FaceResult, classify, download_model, softmax


class ProbabilityTests(unittest.TestCase):
    def test_extreme_logits_are_stable_and_normalized(self):
        result = softmax(np.array([10000, 9999, -10000, 0, 1, 2, 3, 4]))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(float(result.sum()), 1.0, places=6)
        self.assertEqual(int(result.argmax()), 0)

    def test_invalid_model_output_is_rejected(self):
        for scores in (np.zeros(4), np.array([float("nan")] * 8)):
            with self.assertRaises(ValueError):
                softmax(scores)

    def test_original_four_labels_follow_upstream_class_order(self):
        for index, label in ((0, "neutral"), (1, "happy"), (3, "sad"), (4, "angry")):
            scores = np.zeros(8)
            scores[index] = 1
            self.assertEqual(classify(scores, 0.5), (label, 1.0))

    def test_additional_expressions_are_recognized_and_respect_threshold(self):
        for index, label in ((2, "surprise"), (5, "disgust"), (6, "fear"), (7, "contempt")):
            scores = np.zeros(8)
            scores[index] = 1
            self.assertEqual(classify(scores, 0.5), (label, 1.0))
            scores = np.full(8, 0.1)
            scores[index] = 0.3
            self.assertEqual(classify(scores, 0.5), ("uncertain", 0.3))

    def test_low_confidence_is_uncertain(self):
        self.assertEqual(classify(np.full(8, 0.125), 0.5), ("uncertain", 0.125))


class PipelineTests(unittest.TestCase):
    def test_corrupt_cached_weights_are_rejected_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"not a model")
            with patch("emotion.urllib.request.urlopen") as request:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_model(path)
                request.assert_not_called()

    def test_preprocessing_keeps_model_pixel_scale_and_layout(self):
        captured = {}

        class Session:
            def run(self, outputs, inputs):
                captured.update(inputs)
                return [np.zeros((1, 8), dtype=np.float32)]

        recognizer = EmotionRecognizer.__new__(EmotionRecognizer)
        recognizer.session = Session()
        recognizer.input_name = "pixels"
        result = recognizer.predict_crop(np.full((120, 90), 200, dtype=np.uint8))
        tensor = captured["pixels"]
        self.assertEqual(tensor.shape, (1, 1, 64, 64))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(float(tensor.mean()), 200)
        self.assertAlmostEqual(float(result.sum()), 1.0)

    def test_face_coordinates_are_restored_after_detection_resize(self):
        class Detector:
            def detectMultiScale(self, image, **kwargs):
                self_shape = image.shape
                assert self_shape == (360, 640)
                return np.array([[20, 30, 60, 70], [100, 100, 90, 90]])

        recognizer = EmotionRecognizer.__new__(EmotionRecognizer)
        recognizer.detector = Detector()
        recognizer.detector_kind = "haar"
        recognizer.threshold = 0.5
        with patch.object(recognizer, "predict_crop", return_value=np.eye(8)[1]):
            faces = recognizer.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual([f.box for f in faces], [(200, 200, 180, 180), (40, 60, 120, 140)])
        self.assertEqual(faces[0].label, "happy")

    def test_dashboard_handles_no_face_and_small_frames(self):
        frame = np.zeros((64, 80, 3), dtype=np.uint8)
        for faces in (
            [],
            [
                FaceResult(
                    (0, 0, 64, 64),
                    "uncertain",
                    0.125,
                    dict.fromkeys(
                        (
                            "neutral",
                            "happy",
                            "surprise",
                            "sad",
                            "angry",
                            "disgust",
                            "fear",
                            "contempt",
                        ),
                        0.125,
                    ),
                )
            ],
        ):
            rendered = dashboard(frame, faces, 20)
            self.assertEqual(rendered.shape, (720, 1200, 3))
            self.assertEqual(rendered.dtype, np.uint8)

    def test_yunet_preserves_fractional_coordinates_and_clips_negative_origin(self):
        class Detector:
            def setInputSize(self, shape):
                assert shape == (640, 360)

            def detect(self, image):
                assert image.shape == (360, 640, 3)
                row = [-5.25, 10.5, 20, 40, *([0] * 10), 0.9]
                return 1, np.array([row], dtype=np.float32)

        recognizer = EmotionRecognizer.__new__(EmotionRecognizer)
        recognizer.detector = Detector()
        recognizer.detector_kind = "yunet"
        recognizer.threshold = 0.5
        with patch.object(recognizer, "predict_crop", return_value=np.eye(8)[1]):
            result = recognizer.detect(np.zeros((720, 1280, 3), dtype=np.uint8))[0]
        self.assertEqual(result.box, (0, 21, 30, 80))
        self.assertAlmostEqual(result.detection_confidence, 0.9)

    def test_dashboard_renders_every_expression_label(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for label in TARGET_LABELS:
            scores = dict.fromkeys(TARGET_LABELS, 0.0)
            scores[label] = 1.0
            result = FaceResult((10, 10, 80, 80), label, 1.0, scores)
            with patch("dashboard.text") as draw_text:
                rendered = dashboard(frame, [result], 20)
                displayed = [call.args[1] for call in draw_text.call_args_list]
            self.assertEqual(rendered.shape, (720, 1200, 3))
            for name in TARGET_LABELS:
                self.assertIn(name.capitalize(), displayed)
            self.assertIn(label.upper(), displayed)


class CliTests(unittest.TestCase):
    def rejected(self, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(args)

    def test_invalid_options_fail_before_camera_or_output_is_opened(self):
        for args in (
            ["--threshold", "nan"],
            ["--threshold", "1.1"],
            ["--max-frames", "0"],
            ["--camera", "-1"],
            ["--image", "a.jpg", "--video", "b.mp4"],
            ["--headless", "--output", "a.jpg"],
            ["--smoothing", "nan"],
            ["--smoothing", "-1"],
            ["--smoothing", "inf"],
            ["--detector", "invalid"],
            ["--setup", "--check"],
        ):
            self.rejected(args)

    def test_input_cannot_be_overwritten(self):
        self.rejected(["--video", "a.mp4", "--output", "a.mp4"])
        self.rejected(["--image", "a.png", "--jsonl", "a.png"])

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.mp4"
            path.write_bytes(b"keep")
            self.rejected(["--output", str(path)])
            self.assertEqual(path.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
