"""Sustained-expression hooks. Network delivery is explicitly opt-in and bounded."""

import json
import queue
import threading
import urllib.parse
import urllib.request


class EventDetector:
    def __init__(self, label="happy", seconds=2.0, confidence=0.7, cooldown=10.0, hook=None):
        self.label, self.seconds, self.confidence, self.cooldown = (
            label,
            seconds,
            confidence,
            cooldown,
        )
        self.hook = hook
        self.pending: dict = {}
        self.last_sent: dict = {}

    def update(self, faces, timestamp):
        present = set()
        events = []
        for face in faces:
            track_id = face.track_id
            present.add(track_id)
            if (
                face.label != self.label
                or face.confidence < self.confidence
                or face.uncertainty_reason
            ):
                self.pending.pop(track_id, None)
                continue
            if not face.fresh_prediction:
                continue
            since = self.pending.setdefault(track_id, timestamp)
            if (
                timestamp - since >= self.seconds
                and timestamp - self.last_sent.get(track_id, -1e30) >= self.cooldown
            ):
                event = {
                    "type": "sustained_expression",
                    "track_id": track_id,
                    "expression": self.label,
                    "duration_seconds": timestamp - since,
                    "timestamp_seconds": timestamp,
                    "confidence": face.confidence,
                }
                self.last_sent[track_id] = timestamp
                events.append(event)
                if self.hook:
                    self.hook(event)
        self.pending = {key: value for key, value in self.pending.items() if key in present}
        self.last_sent = {key: value for key, value in self.last_sent.items() if key in present}
        return events


class WebhookDispatcher:
    def __init__(self, url: str):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Webhook must be an http(s) URL without credentials.")
        self.url = url
        self.queue: queue.Queue = queue.Queue(maxsize=16)
        self.errors: list[str] = []
        self.closed = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, event):
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.errors.append("Webhook queue full; event dropped")
            self.errors = self.errors[-20:]

    def _worker(self):
        while True:
            event = self.queue.get()
            try:
                if event is None:
                    return
                request = urllib.request.Request(
                    self.url,
                    data=json.dumps(event).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                # Do not forward expression data to an unconfigured redirect destination.
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, headers, newurl):
                        return None

                with urllib.request.build_opener(NoRedirect).open(request, timeout=3) as response:
                    response.read(1024)
            except OSError:
                self.errors.append("Webhook delivery failed")
                self.errors = self.errors[-20:]
            finally:
                self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            # Drain pending work on shutdown, rather than delaying exit for every queued item.
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
            self.queue.put_nowait(None)
            self.thread.join(timeout=4)
