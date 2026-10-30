import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from emotion import RAW_LABELS, FaceResult
from service import create_app


class FakeRecognizer:
    threshold = 0.5
    temperature = 1.0
    auxiliary = None

    def detect(self, frame):
        probabilities = {name: float(name == "happy") for name in RAW_LABELS}
        return [FaceResult((5, 5, 50, 50), "happy", 1, probabilities)]


class ServiceTests(unittest.TestCase):
    def test_api_rejects_unsupported_webhooks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "api.yaml"
            config.write_text("local_only: false\nwebhook: http://127.0.0.1:9999/events\n")
            with self.assertRaisesRegex(ValueError, "does not send webhooks"):
                create_app(config, recognizer_factory=FakeRecognizer)

    def encoded(self):
        return cv2.imencode(".png", np.zeros((80, 80, 3), np.uint8))[1].tobytes()

    def test_rest_and_websocket_return_in_memory_predictions(self):
        with patch.dict(os.environ, {"EXPRESSION_LAB_API_TOKEN": ""}):
            app = create_app(recognizer_factory=FakeRecognizer)
        with TestClient(app) as client:
            self.assertFalse(client.get("/health").json()["model_loaded"])
            result = client.post("/analyze", content=self.encoded())
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["faces"][0]["label"], "happy")
            self.assertEqual(result.json()["storage"], "memory_only")
            with client.websocket_connect("/ws") as socket:
                socket.send_bytes(self.encoded())
                record = socket.receive_json()
                self.assertEqual(record["session"]["frames"], 1)
                self.assertEqual(record["faces"][0]["track_id"], 1)

    def test_invalid_images_fail_cleanly(self):
        with patch.dict(os.environ, {"EXPRESSION_LAB_API_TOKEN": ""}):
            app = create_app(recognizer_factory=FakeRecognizer)
        with TestClient(app) as client:
            self.assertEqual(client.post("/analyze", content=b"not an image").status_code, 422)

    def test_token_and_origin_restrictions(self):
        with patch.dict(os.environ, {"EXPRESSION_LAB_API_TOKEN": "test-token"}):
            app = create_app(recognizer_factory=FakeRecognizer)
        with TestClient(app) as client:
            self.assertEqual(client.post("/analyze", content=self.encoded()).status_code, 403)
            headers = {"Authorization": "Bearer test-token"}
            self.assertEqual(
                client.post("/analyze", content=self.encoded(), headers=headers).status_code, 200
            )
            headers["Origin"] = "https://unrelated.example"
            self.assertEqual(
                client.post("/analyze", content=self.encoded(), headers=headers).status_code, 403
            )
