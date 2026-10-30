import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from events import WebhookDispatcher


class WebhookTests(unittest.TestCase):
    def test_event_delivery_to_explicit_loopback_receiver(self):
        received = []
        delivered = threading.Event()

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()
                delivered.set()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        dispatcher = WebhookDispatcher(f"http://127.0.0.1:{server.server_port}/events")
        event = {"type": "sustained_expression", "expression": "happy", "track_id": 3}
        try:
            dispatcher.submit(event)
            self.assertTrue(delivered.wait(3), "Loopback webhook was not delivered")
            self.assertEqual(received, [event])
        finally:
            dispatcher.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertFalse(dispatcher.thread.is_alive())
        self.assertFalse(dispatcher.errors)
