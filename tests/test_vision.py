import base64
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from expression_lab.app import parse_args, run
from expression_lab.service import create_app
from expression_lab.vision import VisionBusy, VisionClient, VisionWorker, validate_endpoint


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.reply = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "A blue square.",
                                "objects": [{"label": "blue square", "box": [100, 200, 500, 600]}],
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        self.status = 200
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "test-vision"}]}).encode())

            def do_POST(self):
                owner.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(owner.status)
                if owner.status == 302:
                    self.send_header("Location", "/redirected")
                self.end_headers()
                self.wfile.write(json.dumps(owner.reply).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = VisionClient(
            f"http://127.0.0.1:{self.server.server_port}/v1", "test-vision", max_side=224
        )
        self.frame = np.full((400, 800, 3), (255, 0, 0), np.uint8)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_request_contains_resized_image_question_and_no_tools(self):
        self.assertTrue(self.client.check()["ready"])
        with patch.dict(
            os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1"}
        ):
            result = self.client.describe(self.frame, "What color?")
        self.assertEqual(result["answer"], "A blue square.")
        self.assertEqual(result["input_size_wh"], [224, 112])
        self.assertEqual(
            result["objects"], [{"label": "blue square", "box_xyxy": [80, 80, 400, 240]}]
        )
        self.assertEqual(result["frame_size_wh"], [800, 400])
        payload = self.calls[0]
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0]["text"], "What color?")
        encoded = base64.b64decode(content[1]["image_url"]["url"].split(",")[1])
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (112, 224, 3))
        self.assertGreater(decoded[0, 0, 0], 240)
        self.assertNotIn("tools", payload)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")

    def test_text_only_mode_remains_available(self):
        self.client.boxes = False
        self.reply["choices"][0]["message"]["content"] = "A blue square."
        result = self.client.describe(self.frame)
        self.assertEqual(result["answer"], "A blue square.")
        self.assertEqual(result["objects"], [])
        self.assertNotIn("response_format", self.calls[0])

    def test_truncated_grounding_is_rejected(self):
        self.reply["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            self.client.describe(self.frame)

    def test_remote_endpoints_and_redirects_are_rejected(self):
        for url in (
            "https://example.com/v1",
            "http://192.168.0.2:1234/v1",
            "http://localhost:1234/v1",
            "http://user:pass@127.0.0.1:1234/v1",
            "http://127.0.0.1:1234/v1?x=1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_endpoint(url)
        self.status = 302
        with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
            self.client.describe(self.frame)
        self.assertEqual(len(self.calls), 1)

    def test_bad_responses_and_inputs_fail_cleanly(self):
        for reply in ({}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}):
            self.reply = reply
            with self.assertRaisesRegex(RuntimeError, "usable answer"):
                self.client.describe(self.frame)
        with self.assertRaises(ValueError):
            self.client.describe(self.frame, " ")
        with self.assertRaises(ValueError):
            self.client.describe(np.zeros((0, 0, 3), np.uint8))
        self.client.model = "missing"
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.client.check()

    def test_parallel_calls_reject_instead_of_queuing(self):
        with self.client._lock, self.assertRaises(VisionBusy):
            self.client.describe(self.frame)
        self.assertEqual(self.calls, [])

    def test_worker_copies_frame_and_preserves_capture_identity(self):
        entered, release = threading.Event(), threading.Event()
        captured = []
        fake = MagicMock(model="test")

        def describe(frame, question):
            entered.set()
            release.wait(2)
            captured.append(frame[0, 0].copy())
            return {"status": "ok", "answer": question}

        fake.describe.side_effect = describe
        worker = VisionWorker(fake)
        try:
            self.assertTrue(worker.submit(self.frame, "first", 7, 0.5))
            self.assertTrue(entered.wait(1))
            self.frame[:] = 0
            self.assertFalse(worker.submit(self.frame, "second", 8, 0.6))
            self.assertIsNone(worker.poll())
            release.set()
            result = worker.poll(wait=True)
            self.assertEqual(result["source_frame"], 7)
            self.assertEqual(result["source_timestamp_seconds"], 0.5)
            self.assertEqual(result["answer"], "first")
            np.testing.assert_array_equal(captured[0], [255, 0, 0])
        finally:
            release.set()
            worker.close()

    def test_no_storage_forbids_vision_export(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--vision", "--no-storage", "--vision-jsonl", "result.jsonl"])

    def test_desktop_question_editor_preserves_hotkey_letters(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            path = Path(directory) / "input.png"
            cv2.imwrite(str(path), self.frame)
            recognizer = MagicMock(threshold=0.5, temperature=1.0, auxiliary=None)
            recognizer.detect.return_value = []
            worker = MagicMock(pending=None)
            worker.poll.return_value = None
            worker.submit.return_value = True
            stack.enter_context(
                patch("expression_lab.app.EmotionRecognizer", return_value=recognizer)
            )
            stack.enter_context(
                patch("expression_lab.app.client_from_args", return_value=self.client)
            )
            stack.enter_context(patch("expression_lab.app.VisionWorker", return_value=worker))
            stack.enter_context(patch("expression_lab.app.cv2.imshow"))
            stack.enter_context(patch("expression_lab.app.cv2.destroyAllWindows"))
            stack.enter_context(patch("expression_lab.app.cv2.getWindowProperty", return_value=1))
            stack.enter_context(
                patch(
                    "expression_lab.app.cv2.waitKey",
                    side_effect=[ord("t"), *map(ord, "what is q"), 13, ord("q")],
                )
            )
            stack.enter_context(patch.dict(os.environ, {"DISPLAY": ":test"}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            self.assertEqual(run(parse_args(["--vision", "--image", str(path), "--no-storage"])), 0)
            self.assertEqual(worker.submit.call_count, 2)
            self.assertEqual(worker.submit.call_args.args[1], "what is q")
            self.assertEqual(worker.submit.call_args.args[2], 1)

    def test_headless_inference_failure_is_reported_and_returns_nonzero(self):
        self.status = 500
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            cv2.imwrite(str(path), self.frame)
            recognizer = MagicMock(threshold=0.5, temperature=1.0, auxiliary=None)
            recognizer.detect.return_value = []
            with (
                patch("expression_lab.app.EmotionRecognizer", return_value=recognizer),
                patch("expression_lab.app.client_from_args", return_value=self.client),
                contextlib.redirect_stdout(io.StringIO()) as output,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    run(
                        parse_args(["--vision", "--image", str(path), "--headless", "--no-storage"])
                    ),
                    1,
                )
            self.assertEqual(json.loads(output.getvalue())["vision"]["status"], "error")

    def test_headless_image_integrates_vision_without_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            cv2.imwrite(str(path), self.frame)
            recognizer = MagicMock(threshold=0.5, temperature=1.0, auxiliary=None)
            recognizer.detect.return_value = []
            with (
                patch("expression_lab.app.EmotionRecognizer", return_value=recognizer),
                patch("expression_lab.app.client_from_args", return_value=self.client),
                contextlib.redirect_stdout(io.StringIO()) as output,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                args = parse_args(["--vision", "--image", str(path), "--headless", "--no-storage"])
                self.assertEqual(run(args), 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record["faces"], [])
            self.assertEqual(record["vision"]["answer"], "A blue square.")
            self.assertEqual(record["vision"]["source_frame"], 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_vision_check_does_not_open_camera_or_expression_model(self):
        with (
            patch("expression_lab.app.client_from_args", return_value=self.client),
            patch("expression_lab.app.EmotionRecognizer") as model,
            patch("expression_lab.app.cv2.VideoCapture") as camera,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(parse_args(["--vision-check"])), 0)
            model.assert_not_called()
            camera.assert_not_called()

    def test_api_vision_is_opt_in_and_respects_auth(self):
        encoded = cv2.imencode(".png", self.frame)[1].tobytes()
        with patch.dict(os.environ, {"EXPRESSION_LAB_API_TOKEN": "test"}):
            disabled = create_app()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "vision.yaml"
                path.write_text("vision: true\n")
                enabled = create_app(path, vision_factory=lambda: self.client)
        with TestClient(disabled) as client:
            self.assertEqual(
                client.post(
                    "/describe", content=encoded, headers={"Authorization": "Bearer test"}
                ).status_code,
                503,
            )
        with TestClient(enabled) as client:
            self.assertEqual(client.post("/describe", content=encoded).status_code, 403)
            headers = {"Authorization": "Bearer test"}
            response = client.post(
                "/describe?question=What+color%3F", content=encoded, headers=headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "A blue square.")
            with self.client._lock:
                self.assertEqual(
                    client.post("/describe", content=encoded, headers=headers).status_code, 429
                )
            self.assertFalse(client.get("/health").json()["model_loaded"])
            self.assertEqual(
                client.post("/describe", content=b"broken", headers=headers).status_code, 422
            )
            headers["Origin"] = "https://example.com"
            self.assertEqual(
                client.post("/describe", content=encoded, headers=headers).status_code, 403
            )
