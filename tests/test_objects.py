import unittest
from unittest.mock import patch

import cv2
import numpy as np

from expression_lab.dashboard import dashboard, draw_object_boxes
from expression_lab.objects import ObjectOverlay, parse_objects


class ObjectTests(unittest.TestCase):
    def test_coordinate_mapping_rejects_bad_boxes_without_guessing(self):
        data = {
            "answer": "Objects",
            "objects": [
                {"label": "cup", "box": [100, 100, 400, 700]},
                {"label": "backward", "box": [700, 0, 300, 900]},
                {"label": "outside", "box": [-20, 0, 1000, 1000]},
                {"label": "nan", "box": [0, 0, float("nan"), 1000]},
                {"label": "bool", "box": [False, 0, 100, 100]},
                {"label": "", "box": [0, 0, 100, 100]},
            ],
        }
        objects, rejected = parse_objects(data, 200, 100)
        self.assertEqual(objects, [{"label": "cup", "box_xyxy": [20, 10, 80, 70]}])
        self.assertEqual(rejected, 5)
        self.assertEqual(parse_objects({"answer": "nothing", "objects": []}, 200, 100), ([], 0))

    def fixture(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[30:90, 40:100] = np.random.default_rng(42).integers(
            0, 255, (60, 60, 3), dtype=np.uint8
        )
        result = {
            "status": "ok",
            "source_frame": 1,
            "frame_size_wh": [160, 120],
            "objects": [{"label": "box", "box_xyxy": [40, 30, 100, 90]}],
        }
        return frame, result

    def test_delayed_result_replays_motion_to_current_frame(self):
        frame, result = self.fixture()
        overlay = ObjectOverlay()
        overlay.update(frame, 1, 0)
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 6], [0, 1, 4]]), (160, 120))
        overlay.update(shifted, 2, 0.1)
        overlay.accept(result)
        self.assertEqual(len(overlay.objects), 1)
        np.testing.assert_allclose(overlay.objects[0]["box_xyxy"], [46, 34, 106, 94], atol=1)
        self.assertEqual(result["objects"][0]["box_xyxy"], [40, 30, 100, 90])

    def test_cut_blank_and_expired_frames_remove_objects(self):
        frame, result = self.fixture()
        overlay = ObjectOverlay()
        overlay.update(frame, 1, 0)
        overlay.accept(result)
        self.assertEqual(len(overlay.objects), 1)
        overlay.update(np.zeros_like(frame), 2, 0.1)
        self.assertEqual(overlay.objects, [])
        overlay = ObjectOverlay(max_age=1)
        overlay.update(frame, 1, 0)
        overlay.update(frame, 2, 1.1)
        overlay.accept(result)
        self.assertEqual(overlay.objects, [])

    def test_lost_history_and_resolution_change_do_not_reuse_boxes(self):
        frame, result = self.fixture()
        overlay = ObjectOverlay(history_size=2)
        for number in range(1, 4):
            overlay.update(frame, number, number * 0.1)
        overlay.accept(result)
        self.assertEqual(overlay.objects, [])
        overlay.update(np.zeros((80, 80, 3), np.uint8), 4, 0.4)
        overlay.accept(result)
        self.assertEqual(overlay.objects, [])
        self.assertEqual(len(overlay.history), 1)

    def test_names_are_under_boxes_and_clipped_at_image_bottom(self):
        canvas = np.zeros((300, 300, 3), np.uint8)
        with patch("expression_lab.dashboard.text") as text:
            draw_object_boxes(
                canvas, [{"label": "cup", "box_xyxy": [40, 30, 100, 90]}], 0, 0, 300, 300, 300, 300
            )
            self.assertEqual(text.call_args.args[1], "cup")
            self.assertGreater(text.call_args.args[2][1], 90)
            draw_object_boxes(
                canvas,
                [{"label": "book", "box_xyxy": [200, 200, 300, 300]}],
                0,
                0,
                300,
                300,
                300,
                300,
            )
            self.assertLess(text.call_args.args[2][1], 300)
        self.assertTrue(canvas[30, 40].any())

    def test_no_overlays_keeps_input_free_of_object_annotations(self):
        frame, result = self.fixture()
        rendered = dashboard(frame, [], 0, objects=result["objects"], overlays=False)
        np.testing.assert_array_equal(rendered, frame)
