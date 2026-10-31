"""Desktop dashboard drawing and explicit snapshot capture."""

import textwrap
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .emotion import TARGET_LABELS, FaceResult

COLORS = {
    "happy": (121, 220, 112),
    "sad": (239, 166, 96),
    "angry": (103, 111, 245),
    "neutral": (219, 206, 159),
    "surprise": (90, 205, 249),
    "disgust": (143, 185, 95),
    "fear": (226, 157, 206),
    "contempt": (164, 165, 239),
    "uncertain": (169, 171, 182),
}
WINDOW = "Expression Lab"


def text(canvas, value, position, size=0.6, color=(215, 221, 231), thickness=1):
    cv2.putText(
        canvas, value, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA
    )


def dashboard(
    frame: np.ndarray,
    faces: list[FaceResult],
    fps: float,
    selected_id: int | None = None,
    paused=False,
    source="LOCAL CNN",
    *,
    privacy_blur=False,
    overlays=True,
    multi=False,
    histories=None,
    vision=None,
    objects=None,
) -> np.ndarray:
    frame = frame.copy()
    if privacy_blur:
        for face in faces:
            x, y, w, h = face.box
            crop = frame[y : y + h, x : x + w]
            if crop.size:
                tiny = cv2.resize(crop, (max(1, w // 18), max(1, h // 18)))
                frame[y : y + h, x : x + w] = cv2.resize(
                    tiny, (w, h), interpolation=cv2.INTER_NEAREST
                )
    if not overlays:
        return frame
    canvas = np.full((720, 1200, 3), (22, 20, 18), dtype=np.uint8)
    text(canvas, "EXPRESSION LAB", (28, 42), 0.9, (240, 245, 248), 2)
    state = "PAUSED" if paused else source
    text(canvas, f"{state} / FACIAL EXPRESSIONS", (30, 69), 0.42, (153, 163, 170))
    rate = (
        "STILL IMAGE" if source.startswith("IMAGE") else f"{fps:.1f} FPS" if fps > 0 else "-- FPS"
    )
    text(canvas, f"{rate}  |  {len(faces)} FACE(S)", (880, 43), 0.5)

    annotated = frame.copy()
    for face in faces:
        x, y, w, h = face.box
        color = COLORS[face.label]
        cv2.rectangle(
            annotated, (x, y), (x + w, y + h), color, 3 if face.track_id == selected_id else 1
        )
        label = face.label.upper()
        if face.track_id is not None:
            label = f"#{face.track_id} " + label
        if face.label != "uncertain":
            label += f" {face.confidence:.0%}"
        font_scale = max(0.4, min(0.7, frame.shape[1] / 1400))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(th + 8, y)
        cv2.rectangle(
            annotated,
            (x, label_y - th - 8),
            (min(frame.shape[1] - 1, x + tw + 10), label_y + 1),
            color,
            -1,
        )
        text(annotated, label, (x + 5, label_y - 4), font_scale, (22, 20, 18))

    ratio = min(824 / frame.shape[1], 562 / frame.shape[0])
    resized = cv2.resize(annotated, None, fx=ratio, fy=ratio)
    height, width = resized.shape[:2]
    left, top = 28 + (824 - width) // 2, 100 + (562 - height) // 2
    canvas[top : top + height, left : left + width] = resized
    draw_object_boxes(
        canvas, objects or [], left, top, width, height, frame.shape[1], frame.shape[0]
    )

    cv2.rectangle(canvas, (880, 100), (1170, 662), (35, 32, 29), -1)
    text(canvas, "EXPRESSION SCORES", (902, 132), 0.53)
    selected = next(
        (face for face in faces if face.track_id == selected_id), faces[0] if faces else None
    )
    if selected and selected.uncertainty_reason:
        tips = {
            "face_too_small": "Move closer to the camera",
            "too_dark": "Add light in front of your face",
            "overexposed": "Reduce strong light on your face",
            "blurry": "Hold still and check camera focus",
        }
        reasons = (selected.quality or {}).get("reasons", [])
        hint = next(
            (tips[reason] for reason in reasons if reason in tips),
            "Expression unclear; hold a natural pose in good light",
        )
        text(canvas, hint, (30, 91), 0.42, (153, 190, 215))
    text(canvas, f"{len(objects or [])} OBJECT BOX(ES)", (880, 70), 0.42, (153, 163, 170))
    detail = (
        f"Track #{selected.track_id} / TAB to switch"
        if selected and selected.track_id
        else "No face selected"
    )
    text(canvas, detail, (902, 157), 0.40, (153, 163, 170))
    label = selected.label.upper() if selected else "NO FACE"
    text(canvas, label, (902, 213), 0.9, COLORS.get(label.lower(), (169, 171, 182)), 2)
    for index, name in enumerate(TARGET_LABELS):
        y = 250 + index * 44
        value = selected.probabilities[name] if selected else 0.0
        text(canvas, name.capitalize(), (902, y), 0.49)
        text(canvas, f"{value:.0%}", (1101, y), 0.5, COLORS[name])
        cv2.rectangle(canvas, (902, y + 10), (1146, y + 17), (65, 59, 53), -1)
        if value > 0:
            cv2.rectangle(
                canvas, (902, y + 10), (902 + round(244 * value), y + 17), COLORS[name], -1
            )
    text(canvas, "8 expression classes", (902, 602), 0.43, (153, 163, 170))
    text(canvas, "Low confidence = uncertain", (902, 633), 0.40, (153, 163, 170))
    if selected and selected.valence_arousal:
        values = selected.valence_arousal
        cv2.rectangle(canvas, (895, 585), (1156, 615), (35, 32, 29), -1)
        text(canvas, f"V {values['valence']:+.2f}  A {values['arousal']:+.2f}", (902, 602), 0.43)
    if selected and selected.action_units:
        strongest = sorted(selected.action_units.items(), key=lambda item: item[1], reverse=True)[
            :2
        ]
        cv2.rectangle(canvas, (895, 620), (1156, 649), (35, 32, 29), -1)
        text(
            canvas, "  ".join(f"{name} {value:.2f}" for name, value in strongest), (902, 633), 0.38
        )
    if selected and selected.uncertainty_reason == "image_quality":
        cv2.rectangle(canvas, (895, 585), (1156, 615), (35, 32, 29), -1)
        text(canvas, "IMAGE QUALITY REJECTED", (902, 602), 0.37, COLORS["angry"])
    if multi:
        _multi_panel(canvas, faces, histories or {}, selected_id)
    if vision is not None:
        _vision_panel(canvas, vision)
    text(
        canvas,
        "Expression estimates, not a measure of someone's feelings.",
        (30, 696),
        0.47,
        (153, 163, 170),
    )
    controls = (
        "A Ask  R Inspect  T Question  V/M View  Q Quit"
        if vision is not None
        else "SPACE Pause  TAB Face  M View  S Save  Q Quit"
    )
    text(canvas, controls, (685, 696), 0.39)
    return canvas


def draw_object_boxes(canvas, objects, left, top, width, height, source_width, source_height):
    """Draw in the image viewport, with each label just below its rectangle."""
    color = (235, 211, 91)
    for obj in objects:
        x1, y1, x2, y2 = obj["box_xyxy"]
        x1, x2 = [
            left + round(max(0, min(source_width, x)) * width / source_width) for x in (x1, x2)
        ]
        y1, y2 = [
            top + round(max(0, min(source_height, y)) * height / source_height) for y in (y1, y2)
        ]
        x2, y2 = min(left + width - 1, x2), min(top + height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = obj["label"].encode("ascii", "replace").decode()
        size = 0.48
        while (
            len(label) > 1
            and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] > width - 12
        ):
            label = label[:-2] + "~"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        lx = min(x1, left + width - tw - 10)
        # At the image edge keep the name visible against the bottom of its box.
        ly = min(y2 + th + 9, top + height - baseline - 4)
        cv2.rectangle(canvas, (lx, ly - th - 5), (lx + tw + 8, ly + baseline + 3), color, -1)
        text(canvas, label, (lx + 4, ly), size, (22, 20, 18))


def _vision_panel(canvas, state):
    cv2.rectangle(canvas, (880, 100), (1170, 662), (35, 32, 29), -1)
    text(canvas, "LOCAL VISION", (902, 132), 0.62, (121, 220, 112))
    result = state.get("result") or {}
    editing = state.get("editing")
    status = (
        "TYPE / ENTER send / ESC cancel"
        if editing is not None
        else "ANALYZING CAPTURED FRAME..."
        if state.get("busy")
        else "A ask / T question / V switch"
    )
    text(canvas, status, (902, 159), 0.35, (153, 163, 170))
    question = editing + "_" if editing is not None else state.get("question", "")

    # OpenCV Hershey fonts are ASCII; full Unicode remains intact in API/JSON output.
    def lines(value, count):
        value = " ".join(str(value).split()).encode("ascii", "replace").decode()
        return textwrap.wrap(value, width=32, max_lines=count, placeholder="...")

    for i, line in enumerate(lines(question, 3)):
        text(canvas, line, (902, 193 + 20 * i), 0.40, (180, 200, 220))
    if result.get("source_frame") is not None:
        text(
            canvas,
            f"Frame {result['source_frame']} / {result['source_timestamp_seconds']:.1f}s",
            (902, 266),
            0.39,
        )
    answer = (
        result.get("answer")
        or result.get("error")
        or "Show an object, scene or text, then press A."
    )
    for i, line in enumerate(lines(answer, 15)):
        text(canvas, line, (902, 296 + 21 * i), 0.40)
    text(canvas, "May be wrong. Check details.", (902, 643), 0.38, (153, 163, 170))


def _multi_panel(canvas, faces, histories, selected_id):
    cv2.rectangle(canvas, (880, 100), (1170, 662), (35, 32, 29), -1)
    text(canvas, "TRACKED FACES", (902, 132), 0.55)
    text(canvas, "TAB select / M score details", (902, 156), 0.38, (153, 163, 170))
    selected_index = next((i for i, face in enumerate(faces) if face.track_id == selected_id), 0)
    page = selected_index // 4
    for index, face in enumerate(faces[page * 4 : page * 4 + 4]):
        top = 180 + index * 110
        color = COLORS[face.label]
        cv2.rectangle(canvas, (896, top), (1154, top + 95), (54, 49, 44), 1)
        marker = "*" if face.track_id == selected_id else ""
        text(canvas, f"#{face.track_id}{marker} {face.label.upper()}", (905, top + 25), 0.47, color)
        text(canvas, f"{face.confidence:.0%} / recent history", (905, top + 47), 0.39)
        history = list(histories.get(face.track_id, {}).get("history", []))[-50:]
        bar_width = 232 / max(1, len(history))
        for i, (_, label, _) in enumerate(history):
            cv2.rectangle(
                canvas,
                (905 + round(i * bar_width), top + 61),
                (905 + round((i + 1) * bar_width), top + 77),
                COLORS[label],
                -1,
            )
    if not faces:
        text(canvas, "NO FACES", (902, 220), 0.7)
    text(canvas, f"{len(faces)} visible / page {page + 1}", (902, 641), 0.42, (153, 163, 170))


def save_snapshot(directory: Path, rendered: np.ndarray) -> Path:
    """Only called on the user's S key; use an exclusive filename to avoid overwrites."""
    directory.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("Could not encode snapshot.")
    path = directory / f"expression-{uuid4().hex[:12]}.png"
    with path.open("xb") as output:
        output.write(encoded.tobytes())
    return path
