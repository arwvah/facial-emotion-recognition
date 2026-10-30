import math
import unittest

import numpy as np

from emotion import RAW_LABELS, FaceResult
from tracking import ExpressionTracker, select_face


def face(box=(10, 10, 80, 80), winner="happy"):
    scores = dict.fromkeys(RAW_LABELS, 0.0)
    scores[winner] = 1.0
    return FaceResult(box, winner, 1.0, scores)


class TrackingTests(unittest.TestCase):
    def test_smoothing_is_time_based_and_preserves_raw_scores(self):
        tracker = ExpressionTracker(smoothing=0.25)
        first = tracker.update([face()], 0)[0]
        second = tracker.update([face(winner="sad")], 0.1)[0]
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.label, "happy")
        self.assertAlmostEqual(second.probabilities["sad"], 1 - math.exp(-0.1 / 0.25))
        self.assertEqual(second.raw_probabilities["sad"], 1)
        self.assertAlmostEqual(sum(second.probabilities.values()), 1)

    def test_multiple_faces_keep_scores_separate_when_order_changes(self):
        tracker = ExpressionTracker()
        first = tracker.update([face(), face((200, 10, 80, 80), "sad")], 0)
        second = tracker.update([face((201, 10, 80, 80), "sad"), face()], 0.1)
        self.assertEqual([f.track_id for f in second], [first[1].track_id, first[0].track_id])
        self.assertEqual([f.label for f in second], ["sad", "happy"])

    def test_missing_faces_clear_immediately_and_return_with_new_id(self):
        tracker = ExpressionTracker()
        original = tracker.update([face()], 0)[0].track_id
        self.assertEqual(tracker.update([], 0.1), [])
        self.assertEqual(tracker.tracks, {})
        returned = tracker.update([face(winner="angry")], 0.2)[0]
        self.assertNotEqual(returned.track_id, original)
        self.assertEqual(returned.label, "angry")

    def test_distant_face_cannot_inherit_previous_scores(self):
        tracker = ExpressionTracker()
        first = tracker.update([face()], 0)[0]
        second = tracker.update([face((300, 300, 80, 80), "sad")], 0.1)[0]
        self.assertNotEqual(first.track_id, second.track_id)
        self.assertEqual(second.label, "sad")

    def test_matching_is_one_to_one(self):
        tracker = ExpressionTracker()
        tracker.update([face()], 0)
        results = tracker.update([face(), face((11, 11, 80, 80))], 0.1)
        self.assertEqual(len({f.track_id for f in results}), 2)

    def test_long_gaps_and_backward_clock_reset_scores(self):
        for timestamp in (2, -1):
            tracker = ExpressionTracker()
            original = tracker.update([face()], 0)[0]
            updated = tracker.update([face(winner="sad")], timestamp)[0]
            self.assertNotEqual(original.track_id, updated.track_id)
            self.assertEqual(updated.label, "sad")

    def test_disabled_smoothing_uses_current_prediction(self):
        tracker = ExpressionTracker(smoothing=0)
        tracker.update([face()], 0)
        result = tracker.update([face(winner="angry")], 0.01)[0]
        self.assertEqual(result.label, "angry")
        self.assertEqual(result.probabilities, result.raw_probabilities)

    def test_additional_expressions_survive_tracking_and_export(self):
        for label in ("surprise", "disgust", "fear", "contempt"):
            tracker = ExpressionTracker()
            tracker.update([face(winner=label)], 0)
            result = tracker.update([face(winner=label)], 0.1)[0]
            self.assertEqual(result.label, label)
            self.assertEqual(result.probabilities[label], 1)
            self.assertEqual(result.as_dict()["label"], label)
            self.assertEqual(result.as_dict()["raw_probabilities"][label], 1)

    def test_selection_sticks_to_track_and_cycles_visible_faces(self):
        tracker = ExpressionTracker()
        faces = tracker.update([face(), face((300, 10, 80, 80))], 0)
        selected = select_face(faces, None)
        self.assertEqual(select_face(list(reversed(faces)), selected), selected)
        self.assertEqual(select_face(faces, selected, cycle=True), faces[1].track_id)
        self.assertIsNone(select_face([], selected))

    def test_nonfinite_time_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpressionTracker().update([], np.nan)
