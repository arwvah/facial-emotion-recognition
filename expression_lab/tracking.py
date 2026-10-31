"""Short-lived spatial tracks and time-based expression-score smoothing.

Track IDs describe consecutive overlapping boxes, not a person's identity.
Missing detections are removed immediately; no absent face is rendered or retained.
"""

import math
from dataclasses import replace

import numpy as np

from .emotion import RAW_LABELS, FaceResult, classify


def intersection_over_union(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class ExpressionTracker:
    def __init__(self, threshold=0.5, smoothing=0.25, min_overlap=0.3, margin=0.0):
        if not 0 <= threshold <= 1 or not math.isfinite(smoothing) or smoothing < 0:
            raise ValueError("Invalid threshold or smoothing time.")
        if not 0 < min_overlap <= 1:
            raise ValueError("Overlap threshold must be in (0,1].")
        self.threshold = threshold
        if not 0 <= margin <= 1:
            raise ValueError("Expression margin must be in [0,1]")
        self.margin = margin
        self.smoothing = smoothing
        self.min_overlap = min_overlap
        self.tracks = {}
        self.next_id = 1
        self.last_timestamp = None

    def reset(self):
        self.tracks.clear()
        self.last_timestamp = None

    def update(self, faces: list[FaceResult], timestamp: float) -> list[FaceResult]:
        if not math.isfinite(timestamp):
            raise ValueError("Frame timestamp must be finite.")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            self.reset()
        self.last_timestamp = timestamp
        candidates = []
        for index, face in enumerate(faces):
            for track_id, (previous, _, previous_time) in self.tracks.items():
                overlap = intersection_over_union(face.box, previous)
                # Do not blend a returning face after a long time gap.
                if overlap >= self.min_overlap and timestamp - previous_time <= 1.0:
                    candidates.append((overlap, index, track_id))
        matches, used_tracks = {}, set()
        for _, index, track_id in sorted(candidates, reverse=True):
            if index not in matches and track_id not in used_tracks:
                matches[index] = track_id
                used_tracks.add(track_id)
        updated, results = {}, []
        for index, face in enumerate(faces):
            raw = np.array([face.probabilities[label] for label in RAW_LABELS], dtype=np.float32)
            rejected = face.uncertainty_reason == "image_quality"
            if index in matches:
                track_id = matches[index]
                _, old, previous_time = self.tracks[track_id]
                dt = max(0, timestamp - previous_time)
                alpha = -math.expm1(-dt / self.smoothing) if self.smoothing else 1.0
                scores = raw if rejected else alpha * raw + (1 - alpha) * old
            else:
                track_id = self.next_id
                self.next_id += 1
                scores = raw
            label, confidence = classify(scores, self.threshold, self.margin)
            if rejected:
                label = "uncertain"
            updated[track_id] = (face.box, scores, timestamp)
            results.append(
                replace(
                    face,
                    label=label,
                    confidence=confidence,
                    track_id=track_id,
                    probabilities={name: float(value) for name, value in zip(RAW_LABELS, scores)},
                    raw_probabilities=dict(face.probabilities),
                    uncertainty_reason="image_quality"
                    if rejected
                    else "ambiguous_expression"
                    if label == "uncertain" and confidence >= self.threshold
                    else "low_confidence"
                    if label == "uncertain"
                    else None,
                )
            )
        self.tracks = updated
        return results


def select_face(faces: list[FaceResult], selected_id: int | None, cycle=False) -> int | None:
    ids = [face.track_id for face in faces]
    if not ids:
        return None
    if selected_id not in ids:
        return ids[0]
    return ids[(ids.index(selected_id) + 1) % len(ids)] if cycle else selected_id
