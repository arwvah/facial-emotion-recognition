"""Observable image-quality measurements and five-landmark eye-line alignment."""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityPolicy:
    min_face_size: int = 48
    min_brightness: float = 25.0
    max_brightness: float = 240.0
    min_sharpness: float = 20.0

    def measure(self, crop: np.ndarray) -> dict:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        reasons = []
        if min(gray.shape) < self.min_face_size:
            reasons.append("face_too_small")
        if brightness < self.min_brightness:
            reasons.append("too_dark")
        if brightness > self.max_brightness:
            reasons.append("overexposed")
        if sharpness < self.min_sharpness:
            reasons.append("blurry")
        return {
            "accepted": not reasons,
            "reasons": reasons,
            "brightness": brightness,
            "sharpness": sharpness,
            "min_dimension": min(gray.shape),
        }


def aligned_crop(frame: np.ndarray, box, landmarks=None, size=128) -> tuple[np.ndarray, bool]:
    """Place eyes at consistent positions, preserving aspect ratio and removing head roll."""
    x, y, w, h = box
    if landmarks is not None:
        points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        eyes = points[:2][np.argsort(points[:2, 0])]
        left, right = eyes
        delta = right - left
        distance = float(np.linalg.norm(delta))
        # Reject degenerate landmarks and excessive rotations instead of making a bad warp.
        if np.isfinite(points).all() and distance > 5 and abs(float(delta[1])) < distance * 0.9:
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
            center = (left + right) / 2
            transform = cv2.getRotationMatrix2D(
                tuple(map(float, center)), angle, size * 0.36 / distance
            )
            transform[:, 2] += np.array([size * 0.5, size * 0.36]) - center
            crop = cv2.warpAffine(
                frame,
                transform,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return crop, True
    return frame[y : y + h, x : x + w].copy(), False


def augment(crop: np.ndarray, enabled: bool) -> list[np.ndarray]:
    if not enabled:
        return [crop]
    h, w = crop.shape[:2]
    margin = max(1, round(min(h, w) * 0.04))
    inner = crop[margin : h - margin, margin : w - margin]
    return [crop, cv2.flip(crop, 1), inner if inner.size else crop]
