import unittest
from unittest.mock import MagicMock

import numpy as np

from pipeline import FramePipeline, InferenceWorker


class PipelineOptionsTests(unittest.TestCase):
    def test_worker_propagates_exceptions_and_bounds_pending_work(self):
        pipeline = MagicMock()
        pipeline.process.side_effect = ValueError("inference failed")
        worker = InferenceWorker(pipeline)
        try:
            worker.submit(np.zeros((10, 10, 3), np.uint8), 0)
            with self.assertRaises(RuntimeError):
                worker.submit(np.zeros((10, 10, 3), np.uint8), 1)
            with self.assertRaisesRegex(ValueError, "inference"):
                worker.result()
        finally:
            worker.close()

    def test_detection_skipping_never_propagates_faces_into_a_blank_frame(self):
        recognizer = MagicMock(threshold=0.5)
        recognizer.threshold = 0.5
        pipeline = FramePipeline(recognizer, detect_every=3, infer_every=3)
        pipeline.previous_gray = np.zeros((100, 100), np.uint8)
        from tests.test_tracking import face

        pipeline.previous_observations = [face(box=(10, 10, 60, 60))]
        self.assertEqual(pipeline._propagate(np.zeros((100, 100, 3), np.uint8)), [])
