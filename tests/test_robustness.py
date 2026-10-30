import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app import parse_args
from backends import scale_probabilities
from configuration import apply_calibration
from emotion import RAW_LABELS, EmotionRecognizer, FaceResult
from quality import QualityPolicy, aligned_crop, augment
from tracking import ExpressionTracker


class QualityTests(unittest.TestCase):
    def test_dark_blurry_small_crop_is_rejected_with_separate_reasons(self):
        result = QualityPolicy().measure(np.zeros((20, 20, 3), np.uint8))
        self.assertEqual(set(result["reasons"]), {"too_dark", "blurry", "face_too_small"})

    def test_good_crop_passes_and_overexposed_crop_is_flagged(self):
        random = np.random.default_rng(1).integers(30, 220, (80, 80, 3), dtype=np.uint8)
        self.assertTrue(QualityPolicy().measure(random)["accepted"])
        self.assertIn(
            "overexposed", QualityPolicy().measure(np.full((80, 80), 255, np.uint8))["reasons"]
        )

    def test_alignment_places_tilted_eyes_at_equal_y_and_consistent_scale(self):
        frame = np.zeros((240, 240, 3), np.uint8)
        eyes = np.array([[70, 80], [150, 120]], np.float32)
        for x, y in eyes:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)
        crop, aligned = aligned_crop(
            frame, (30, 30, 170, 180), [*eyes.tolist(), [120, 130], [85, 170], [150, 170]]
        )
        self.assertTrue(aligned)
        self.assertEqual(crop.shape, (128, 128, 3))
        left = cv2.minMaxLoc(crop[:, :64, 0])[3]
        right = cv2.minMaxLoc(crop[:, 64:, 0])[3]
        self.assertLessEqual(abs(left[1] - right[1]), 2)
        self.assertLessEqual(abs((right[0] + 64 - left[0]) - 128 * 0.36), 4)

    def test_degenerate_landmarks_fall_back_without_warp(self):
        frame = np.full((100, 100, 3), 100, np.uint8)
        crop, aligned = aligned_crop(frame, (10, 10, 40, 50), [[20, 20]] * 5)
        self.assertFalse(aligned)
        self.assertEqual(crop.shape, (50, 40, 3))
        self.assertEqual(len(augment(crop, True)), 3)

    def test_skip_quality_never_runs_classifier_or_inherits_a_confident_track(self):
        recognizer = EmotionRecognizer.__new__(EmotionRecognizer)
        recognizer.threshold = 0.5
        recognizer.quality_policy = QualityPolicy()
        recognizer.quality_mode = "skip"
        observation = FaceResult((10, 10, 70, 70), "uncertain", 0, dict.fromkeys(RAW_LABELS, 0.125))
        with patch.object(recognizer, "predict_crop") as predict:
            result = recognizer.classify_observations(
                np.zeros((100, 100, 3), np.uint8), [observation]
            )[0]
            predict.assert_not_called()
        tracker = ExpressionTracker()
        good = FaceResult(
            observation.box, "happy", 1, {label: float(label == "happy") for label in RAW_LABELS}
        )
        tracker.update([good], 0)
        tracked = tracker.update([result], 0.1)[0]
        self.assertEqual(tracked.label, "uncertain")
        self.assertEqual(tracked.uncertainty_reason, "image_quality")
        self.assertTrue(tracked.classification_skipped)


class ConfigurationTests(unittest.TestCase):
    def test_yaml_paths_are_relative_and_cli_overrides_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "settings.yaml"
            config.write_text("threshold: 0.7\nalign: false\nimage: input.png\n")
            options = parse_args(["--config", str(config), "--threshold", "0.6"])
            self.assertEqual(options.threshold, 0.6)
            self.assertFalse(options.align)
            self.assertEqual(options.image, Path(directory) / "input.png")

    def test_invalid_settings_and_no_storage_exports_are_rejected(self):
        cases = (
            ["--no-storage", "--report", "a.html"],
            ["--webhook", "https://example.com/hook"],
            ["--detect-every", "0"],
            ["--temperature", "0"],
            ["--min-brightness", "250", "--max-brightness", "20"],
        )
        for case in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(case)

    def test_unknown_yaml_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.yaml"
            config.write_text("threshhold: 0.5")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config", str(config)])

    def test_calibration_is_bound_to_model_and_preprocessing_contract(self):
        spec = SimpleNamespace(sha256="abc", signature=lambda: {"model": "contract"})
        recognizer = SimpleNamespace(backend=SimpleNamespace(spec=spec), temperature=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temperature.json"
            data = {
                "temperature": 2,
                "model_sha256": "abc",
                "model_contract": spec.signature(),
                "preprocessing": {"align": True, "tta": False},
            }
            path.write_text(json.dumps(data))
            args = SimpleNamespace(calibration=path, align=True, tta=False)
            apply_calibration(recognizer, args)
            self.assertEqual(recognizer.temperature, 2)
            args.tta = True
            with self.assertRaises(ValueError):
                apply_calibration(recognizer, args)

    def test_temperature_softens_scores_without_inventing_missing_classes(self):
        probabilities = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0])
        scaled = scale_probabilities(probabilities, 2)
        self.assertAlmostEqual(float(scaled[0]), 0.75)
        self.assertEqual(float(scaled[2]), 0)
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                scale_probabilities(probabilities, value)
