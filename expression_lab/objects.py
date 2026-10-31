"""Validated model object coordinates and bounded optical-flow display tracking."""

import math
from collections import deque

import cv2
import numpy as np

OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "objects": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "maxLength": 48},
                    "box": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["label", "box"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "objects"],
    "additionalProperties": False,
}
OBJECT_PROMPT = (
    " Return a JSON object with answer (a concise answer to the question) and objects. "
    "Locate up to 6 main visible objects, even if the question is about something else. "
    "Each object has a short label (object name) and box [left, top, right, bottom]. "
    "Use tight bounding boxes with integer coordinates normalized to 0..1000 relative to "
    "the whole image; top-left is (0,0), bottom-right is (1000,1000). "
    "Do not box the whole background. If no object can be located, return objects: []."
)


def parse_objects(data, width, height):
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("answer"), str)
        or not isinstance(data.get("objects"), list)
    ):
        raise TypeError("Expected a JSON answer and object list")
    objects = []
    rejected = 0
    for item in data["objects"][:6]:
        if not isinstance(item, dict):
            rejected += 1
            continue
        label, box = item.get("label"), item.get("box")
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > 48
            or not isinstance(box, list)
            or len(box) != 4
        ):
            rejected += 1
            continue
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1000
            for v in box
        ):
            rejected += 1
            continue
        left, top, right, bottom = box
        if left >= right or top >= bottom:
            rejected += 1
            continue
        pixels = [
            round(left * width / 1000),
            round(top * height / 1000),
            round(right * width / 1000),
            round(bottom * height / 1000),
        ]
        if pixels[2] - pixels[0] < 2 or pixels[3] - pixels[1] < 2:
            rejected += 1
            continue
        label = " ".join(label.split())
        objects.append({"label": label, "box_xyxy": pixels})
    return objects, rejected + max(0, len(data["objects"]) - 6)


class ObjectOverlay:
    """Replay recent motion for delayed results; drop lost, expired or unmatchable boxes."""

    def __init__(self, history_size=180, max_age=5.0):
        self.history: deque = deque(maxlen=history_size)
        self.objects: list[dict] = []
        self.source_time = -float("inf")
        self.max_age = max_age
        self.size = None

    def update(self, frame, frame_number, timestamp):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = min(1, 384 / max(w, h))
        gray = cv2.resize(gray, (max(1, round(w * ratio)), max(1, round(h * ratio))))
        if self.size != (w, h) or (self.history and timestamp < self.history[-1][1]):
            self.history.clear()
            self.objects = []
        self.size = (w, h)
        if self.history and self.objects:
            self.objects = self._move(self.history[-1][2], gray, self.objects)
        if timestamp - self.source_time > self.max_age:
            self.objects = []
        self.history.append((frame_number, timestamp, gray))

    def accept(self, result):
        self.objects = []
        if (
            not self.history
            or result.get("status") != "ok"
            or result.get("frame_size_wh") != list(self.size or ())
        ):
            return
        frames = list(self.history)
        index = next(
            (i for i, entry in enumerate(frames) if entry[0] == result.get("source_frame")), None
        )
        if index is None:
            return
        self.source_time = frames[index][1]
        if frames[-1][1] - self.source_time > self.max_age:
            return
        self.objects = result.get("objects", [])
        for i in range(index + 1, len(frames)):
            self.objects = self._move(frames[i - 1][2], frames[i][2], self.objects)
            if not self.objects:
                break

    def _move(self, previous, current, objects):
        if previous.shape != current.shape or self.size is None:
            return []
        # Unchanged paused/static frames need no features (uniform objects may lack corners).
        if np.array_equal(previous, current):
            return objects
        gh, gw = current.shape
        width, height = self.size
        sx, sy = gw / width, gh / height
        moved_objects = []
        for obj in objects:
            x1, y1, x2, y2 = obj["box_xyxy"]
            x, y = max(0, round(x1 * sx)), max(0, round(y1 * sy))
            right, bottom = min(gw, round(x2 * sx)), min(gh, round(y2 * sy))
            if right - x < 3 or bottom - y < 3:
                continue
            mask = np.zeros_like(previous)
            mask[y:bottom, x:right] = 255
            points = cv2.goodFeaturesToTrack(previous, 30, 0.01, 3, mask=mask)
            if points is None or len(points) < 3:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, points.copy())
            if moved is None or status is None:
                continue
            back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                current, previous, moved, moved.copy()
            )
            if back is None or reverse_status is None:
                continue
            valid = (
                (status.ravel() == 1)
                & (reverse_status.ravel() == 1)
                & (np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.5)
            )
            if valid.sum() < 3:
                continue
            dx, dy = np.median((moved - points)[valid].reshape(-1, 2), axis=0)
            if (
                not np.isfinite([dx, dy]).all()
                or math.hypot(dx, dy) > max(right - x, bottom - y) * 0.5
            ):
                continue
            nx, ny = round(x + float(dx)), round(y + float(dy))
            bw, bh = right - x, bottom - y
            if nx < 0 or ny < 0 or nx + bw > gw or ny + bh > gh:
                continue
            before, after = previous[y:bottom, x:right], current[ny : ny + bh, nx : nx + bw]
            if after.std() < 2 or np.abs(before.astype(float) - after).mean() > 35:
                continue
            coords = [
                round(x1 + dx / sx),
                round(y1 + dy / sy),
                round(x2 + dx / sx),
                round(y2 + dy / sy),
            ]
            if coords[0] < 0 or coords[1] < 0 or coords[2] > width or coords[3] > height:
                continue
            moved_objects.append({**obj, "box_xyxy": coords})
        return moved_objects
