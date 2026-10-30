import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app import main, parse_args, run
from dashboard import save_snapshot
from emotion import RAW_LABELS, FaceResult


class PlaybackTests(unittest.TestCase):
    def test_no_storage_disables_snapshot_key(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            source = Path(directory) / "input.png"
            self.assertTrue(cv2.imwrite(str(source), np.zeros((80, 80, 3), np.uint8)))
            recognizer = MagicMock()
            recognizer.threshold = 0.5
            recognizer.temperature = 1.0
            recognizer.auxiliary = None
            recognizer.detect.return_value = []
            stack.enter_context(patch("app.EmotionRecognizer", return_value=recognizer))
            stack.enter_context(patch("app.cv2.imshow"))
            stack.enter_context(patch("app.cv2.destroyAllWindows"))
            stack.enter_context(patch("app.cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("app.cv2.waitKey", side_effect=[ord("s"), ord("q")]))
            stack.enter_context(patch.dict("os.environ", {"DISPLAY": ":test"}))
            snapshot = stack.enter_context(patch("app.save_snapshot"))
            error = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(run(parse_args(["--image", str(source), "--no-storage"])), 0)
            snapshot.assert_not_called()
            self.assertIn("Snapshot disabled", error.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_check_never_opens_a_camera(self):
        with (
            patch("diagnostics.system_report", return_value={"ready": True}),
            patch("app.cv2.VideoCapture") as capture,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(parse_args(["--check"])), 0)
            capture.assert_not_called()

    def test_missing_display_fails_cleanly_before_camera_initialization(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("app.sys.platform", "linux"),
            patch("app.cv2.VideoCapture") as capture,
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            self.assertEqual(main([]), 1)
            self.assertIn("--headless", error.getvalue())
            capture.assert_not_called()

    def test_pausing_never_advances_capture_or_duplicates_exports(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            source = Path(directory) / "input.mp4"
            source.touch()
            output = Path(directory) / "records.jsonl"
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.get.return_value = 25
            capture.read.return_value = (True, np.zeros((120, 160, 3), dtype=np.uint8))
            recognizer = MagicMock()
            recognizer.threshold = 0.5
            recognizer.temperature = 1.0
            recognizer.auxiliary = None
            scores = dict.fromkeys(RAW_LABELS, 0.0)
            scores["happy"] = 1.0
            recognizer.detect.return_value = [FaceResult((10, 10, 80, 80), "happy", 1, scores)]
            stack.enter_context(patch("app.EmotionRecognizer", return_value=recognizer))
            stack.enter_context(patch("app.cv2.VideoCapture", return_value=capture))
            stack.enter_context(patch("app.cv2.imshow"))
            stack.enter_context(patch("app.cv2.destroyAllWindows"))
            stack.enter_context(patch("app.cv2.getWindowProperty", return_value=1))
            # Process one frame, pause for two UI polls, resume and process the next.
            stack.enter_context(patch("app.cv2.waitKey", side_effect=[32, -1, 32, ord("q")]))
            stack.enter_context(patch.dict("os.environ", {"DISPLAY": ":test"}))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertEqual(run(parse_args(["--video", str(source), "--jsonl", str(output)])), 0)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(capture.read.call_count, 2)
            self.assertEqual(recognizer.detect.call_count, 2)
            self.assertEqual([r["timestamp_seconds"] for r in records], [0, 1 / 25])
            self.assertEqual(records[0]["faces"][0]["track_id"], records[1]["faces"][0]["track_id"])
            capture.release.assert_called_once()

    def test_snapshot_files_are_unique_and_decodable(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((90, 120, 3), dtype=np.uint8)
            folder = Path(directory) / "snapshots"
            first = save_snapshot(folder, image)
            second = save_snapshot(folder, image)
            self.assertNotEqual(first, second)
            np.testing.assert_array_equal(cv2.imread(str(first)), image)

    def test_snapshot_failure_does_not_overwrite_an_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            with patch("dashboard.uuid4") as identifier:
                identifier.return_value.hex = "abcdef123456"
                first = save_snapshot(Path(directory), image)
                original = first.read_bytes()
                with self.assertRaises(FileExistsError):
                    save_snapshot(Path(directory), image)
                self.assertEqual(first.read_bytes(), original)
