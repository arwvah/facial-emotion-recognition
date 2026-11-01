import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from expression_lab.age import AgeEstimator
from expression_lab.emotion import RAW_LABELS, EmotionRecognizer, FaceResult
from expression_lab.quality import QualityPolicy
from scripts.train_age_head import reference_split_map


class AgeTests(unittest.TestCase):
    def test_reference_splits_remove_duplicates_and_preserve_record_ids(self):
        records = [{"source_split": "train", "index": 0, "sha256": "training"}]
        records += [
            {"source_split": "valid", "index": i, "sha256": f"validation-{i}"} for i in range(6)
        ]
        records.append({"source_split": "valid", "index": 6, "sha256": "training"})
        mapping = reference_split_map(records)
        self.assertEqual(mapping[("train", 0)], "train")
        self.assertNotIn(("valid", 6), mapping)
        self.assertTrue(all(mapping[("valid", i)] in ("val", "test") for i in range(6)))
        self.assertEqual(mapping, reference_split_map(records))

    def test_age_backend_rejects_invalid_and_clamps_extremes(self):
        model = AgeEstimator.__new__(AgeEstimator)
        model.backend = MagicMock()
        model.backend.run.return_value = np.array([[200.0]])
        self.assertEqual(model.predict(np.zeros((80, 80, 3), np.uint8)), 116)
        model.backend.run.return_value = np.array([[np.nan]])
        with self.assertRaisesRegex(ValueError, "finite"):
            model.predict(np.zeros((80, 80, 3), np.uint8))
        model.backend.run.return_value = np.array([[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "one"):
            model.predict(np.zeros((80, 80, 3), np.uint8))

    def test_bad_quality_skips_age_inference(self):
        model = EmotionRecognizer.__new__(EmotionRecognizer)
        model.threshold = 0.5
        model.quality_policy = QualityPolicy()
        model.quality_mode = "skip"
        model.age_estimator = MagicMock()
        face = FaceResult((10, 10, 50, 50), "uncertain", 0, dict.fromkeys(RAW_LABELS, 0.125))
        with patch.object(model, "predict_crop"):
            result = model.classify_observations(np.zeros((80, 80, 3), np.uint8), [face])[0]
        model.age_estimator.predict.assert_not_called()
        self.assertIsNone(result.as_dict()["age_estimate_years"])
        self.assertEqual(result.uncertainty_reason, "image_quality")
