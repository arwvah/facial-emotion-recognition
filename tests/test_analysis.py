import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analytics import SessionAnalytics
from emotion import RAW_LABELS, FaceResult
from evaluation import fairness_metrics, fit_temperature, metrics, read_manifest
from events import EventDetector


def sample(label="happy", confidence=1, track_id=1):
    scores = {name: (confidence if name == label else (1 - confidence) / 7) for name in RAW_LABELS}
    return FaceResult((0, 0, 60, 60), label, confidence, scores, track_id=track_id)


class AnalysisTests(unittest.TestCase):
    def test_skipped_quality_frame_breaks_expression_transition(self):
        analytics = SessionAnalytics()
        analytics.update([sample("happy")], 0)
        rejected = sample("uncertain")
        rejected.fresh_prediction = False
        rejected.uncertainty_reason = "image_quality"
        analytics.update([rejected], 0.1)
        analytics.update([sample("sad")], 0.2)
        track = analytics.summary()["tracks"][0]
        self.assertEqual(track["transition_counts"], {})
        self.assertEqual(track["fresh_samples"], 2)

    def test_timelines_are_bounded_and_transitions_exclude_uncertain_gaps(self):
        analytics = SessionAnalytics(history_size=2)
        for index, label in enumerate(["happy", "happy", "sad", "uncertain", "angry"]):
            analytics.update([sample(label)], index / 10)
        track = analytics.summary()["tracks"][0]
        self.assertEqual(len(track["timeline"]), 2)
        self.assertEqual(track["fresh_samples"], 5)
        self.assertEqual(track["dominant_expression"], "happy")
        self.assertEqual(track["transition_counts"], {"happy -> sad": 1})

    def test_reports_are_standalone_and_protect_existing_files(self):
        analytics = SessionAnalytics()
        analytics.update([sample()], 0)
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / "report.html"
            analytics.write_report(html)
            text = html.read_text()
            self.assertIn("<svg", text)
            self.assertNotIn("<script", text)
            self.assertNotIn("http", text)
            with self.assertRaises(FileExistsError):
                analytics.write_report(html)
            path = Path(directory) / "report.json"
            analytics.write_report(path)
            self.assertEqual(json.loads(path.read_text())["frames"], 1)

    def test_events_require_sustained_confidence_and_reset_on_disappearance(self):
        captured = []
        detector = EventDetector(seconds=1, cooldown=5, hook=captured.append)
        self.assertEqual(detector.update([sample()], 0), [])
        self.assertEqual(detector.update([sample()], 0.9), [])
        self.assertEqual(len(detector.update([sample()], 1)), 1)
        self.assertEqual(detector.update([sample()], 2), [])
        detector.update([], 3)
        self.assertEqual(detector.update([sample()], 4), [])
        self.assertEqual(len(captured), 1)

    def test_quality_rejection_resets_event_timer(self):
        detector = EventDetector(seconds=1)
        detector.update([sample()], 0)
        rejected = sample()
        rejected.uncertainty_reason = "image_quality"
        rejected.fresh_prediction = False
        detector.update([rejected], 0.7)
        self.assertEqual(detector.update([sample()], 1.1), [])

    def test_confusion_counts_uncertainty_as_a_missed_prediction(self):
        probabilities = np.eye(8)[[0, 1, 2]].astype(float)
        probabilities[2] = np.full(8, 0.125)
        result = metrics([0, 1, 2], probabilities, 0.5)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["confusion_matrix"][2][8], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)

    def test_temperature_fit_improves_overconfident_calibration_loss(self):
        probabilities = np.tile(np.array([0.99, 0.01, 0, 0, 0, 0, 0, 0]), (4, 1))
        temperature, before, after = fit_temperature(probabilities, [0, 0, 1, 1])
        self.assertGreater(temperature, 1)
        self.assertLess(after, before)

    def test_fairness_does_not_claim_results_for_unlabeled_or_small_groups(self):
        p = np.eye(8)[[0, 1]]
        result = fairness_metrics([{}, {}], [0, 1], p, 0.5, [False, False])
        self.assertEqual(result["status"], "not_assessed_no_group_annotations")
        small = fairness_metrics([{"group": "A"}, {"group": "B"}], [0, 1], p, 0.5, [False, False])
        self.assertEqual(small["groups"]["A"]["status"], "insufficient_samples")
        self.assertIsNone(small["accuracy_gap"])

    def test_manifest_rejects_subject_leakage_between_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png"):
                (root / name).touch()
            path = root / "data.csv"
            path.write_text(
                "path,label,split,subject_id\na.png,happy,calibration,person\nb.png,sad,test,person\n"
            )
            with self.assertRaisesRegex(ValueError, "Subject"):
                read_manifest(path)
