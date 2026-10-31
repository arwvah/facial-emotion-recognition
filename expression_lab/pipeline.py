"""Optical-flow propagation between detector passes and bounded asynchronous processing."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np

from .tracking import ExpressionTracker


class FramePipeline:
    def __init__(self, recognizer, smoothing=0.25, min_overlap=0.3, detect_every=1, infer_every=1):
        self.recognizer = recognizer
        self.tracker = ExpressionTracker(recognizer.threshold, smoothing, min_overlap)
        margin = getattr(recognizer, "expression_margin", 0.0)
        self.tracker.margin = margin if isinstance(margin, (int, float)) else 0.0
        self.detect_every, self.infer_every = detect_every, infer_every
        self.index = 0
        self.previous_gray = None
        self.previous_faces = []
        self.previous_observations = []
        self.last_detection = False
        self.last_inference = False

    def _propagate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None or self.previous_gray.shape != gray.shape:
            return []
        observations = []
        height, width = gray.shape
        for previous in self.previous_observations:
            x, y, w, h = previous.box
            mask = np.zeros_like(gray)
            mask[y : y + h, x : x + w] = 255
            points = cv2.goodFeaturesToTrack(self.previous_gray, 25, 0.01, 4, mask=mask)
            if points is None or len(points) < 3:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray, points, points.copy()
            )
            if moved is None or status is None:
                continue
            valid = status.reshape(-1) == 1
            if valid.sum() < 3:
                continue
            movement = np.median((moved - points)[valid].reshape(-1, 2), axis=0)
            if not np.isfinite(movement).all() or np.linalg.norm(movement) > max(w, h) * 0.5:
                continue
            nx, ny = round(x + float(movement[0])), round(y + float(movement[1]))
            # An empty/very different crop means the face left; do not propagate stale boxes.
            if nx < 0 or ny < 0 or nx + w > width or ny + h > height:
                continue
            old_crop = self.previous_gray[y : y + h, x : x + w]
            new_crop = gray[ny : ny + h, nx : nx + w]
            if (
                float(new_crop.std()) < 2
                or float(np.abs(old_crop.astype(float) - new_crop).mean()) > 55
            ):
                continue
            landmarks = (
                (np.asarray(previous.landmarks) + movement).tolist() if previous.landmarks else None
            )
            observations.append(
                replace(
                    previous, box=(nx, ny, w, h), landmarks=landmarks, detection_confidence=None
                )
            )
        return observations

    def process(self, frame, timestamp):
        # The default path retains the original recognizer contract for external integrations.
        if self.detect_every == self.infer_every == 1:
            faces = self.recognizer.detect(frame)
            self.last_detection = self.last_inference = True
        else:
            self.last_detection = (
                self.index % self.detect_every == 0 or not self.previous_observations
            )
            observations = (
                self.recognizer.locate(frame) if self.last_detection else self._propagate(frame)
            )
            self.last_inference = self.index % self.infer_every == 0 or self.last_detection
            if self.last_inference:
                faces = self.recognizer.classify_observations(frame, observations)
            else:
                from .quality import QualityPolicy
                from .tracking import intersection_over_union

                faces = []
                for observation in observations:
                    matches = [
                        (intersection_over_union(observation.box, old.box), old)
                        for old in self.previous_faces
                    ]
                    overlap, old = max(matches, key=lambda item: item[0]) if matches else (0, None)
                    if old is None or overlap < 0.3:
                        continue
                    x, y, w, h = observation.box
                    quality = getattr(self.recognizer, "quality_policy", QualityPolicy()).measure(
                        frame[y : y + h, x : x + w]
                    )
                    if not quality["accepted"]:
                        faces.append(
                            replace(
                                observation,
                                quality=quality,
                                label="uncertain",
                                uncertainty_reason="image_quality",
                                fresh_prediction=False,
                            )
                        )
                    else:
                        faces.append(
                            replace(
                                old,
                                box=observation.box,
                                landmarks=observation.landmarks,
                                quality=quality,
                                fresh_prediction=False,
                            )
                        )
            self.previous_observations = observations
        tracked = self.tracker.update(faces, timestamp)
        self.previous_faces = tracked
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.index += 1
        return tracked


class InferenceWorker:
    """One worker and at most one prefetched result, without concurrent model access."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expression-inference")
        self.pending = None

    def submit(self, frame, timestamp):
        if self.pending is not None:
            raise RuntimeError("Consume the pending inference before submitting another frame")
        self.pending = self.executor.submit(self.pipeline.process, frame.copy(), timestamp)

    def result(self):
        if self.pending is None:
            raise RuntimeError("No pending inference")
        future, self.pending = self.pending, None
        return future.result()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
