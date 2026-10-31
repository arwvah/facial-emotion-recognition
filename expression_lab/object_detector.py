"""Local NanoDet-Plus detection with the OpenCV Zoo model contract (Apache-2.0)."""

import cv2
import numpy as np
import onnxruntime as ort

COCO_LABELS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def letterbox(frame, size=416):
    height, width = frame.shape[:2]
    ratio = size / max(height, width)
    nw, nh = max(1, int(width * ratio)), max(1, int(height * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    image = np.zeros((size, size, 3), np.uint8)
    image[top : top + nh, left : left + nw] = resized
    return image, (left, top, nw / width, nh / height)


def decode_outputs(outputs, threshold=0.4):
    if len(outputs) != 6:
        raise ValueError("NanoDet requires six output tensors")
    boxes, probabilities = [], []
    # The pinned ONNX graph returns three class heads, then three box heads.
    for stride, raw_scores, raw_distance in zip((8, 16, 32), outputs[:3], outputs[3:]):
        scores = np.asarray(raw_scores).reshape(-1, 80)
        logits = np.asarray(raw_distance).reshape(-1, 4, 8)
        if not np.isfinite(scores).all() or not np.isfinite(logits).all():
            raise ValueError("Nonfinite NanoDet output")
        size = 416 // stride
        gx, gy = np.meshgrid(np.arange(size), np.arange(size))
        centers = np.column_stack((gx.ravel(), gy.ravel())) * stride + (stride - 1) * 0.5
        if len(centers) != len(scores) or len(scores) != len(logits):
            raise ValueError("NanoDet output shape differs from its pinned contract")
        keep = np.where(scores.max(axis=1) >= threshold)[0]
        logits, scores, centers = logits[keep], scores[keep], centers[keep]
        distribution = np.exp(logits - logits.max(axis=2, keepdims=True))
        distribution /= distribution.sum(axis=2, keepdims=True)
        distances = (distribution * np.arange(8)).sum(axis=2) * stride
        boxes.append(np.column_stack((centers - distances[:, :2], centers + distances[:, 2:])))
        probabilities.append(scores)
    return np.concatenate(boxes), np.concatenate(probabilities)


class ObjectDetector:
    def __init__(self, threshold=0.4):
        from .emotion import checksum
        from .recognition_models import NANODET_PATH, NANODET_SHA256

        if not 0 < threshold <= 1:
            raise ValueError("Object confidence threshold must be in (0,1]")
        if not NANODET_PATH.is_file():
            raise FileNotFoundError("Object detector missing. Run ./run.sh --setup-recognition")
        if checksum(NANODET_PATH) != NANODET_SHA256:
            raise ValueError("Object detector checksum mismatch")
        self.threshold = threshold
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        # This fixed older export lists weights as overridable initializers; do not
        # print its ~150 optimization warnings on every launch. Errors remain visible.
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(NANODET_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        image, (left, top, sx, sy) = letterbox(frame)
        values = (
            image.astype(np.float32) - np.array([103.53, 116.28, 123.675], np.float32)
        ) / np.array([57.375, 57.12, 58.395], np.float32)
        tensor = np.ascontiguousarray(values.transpose(2, 0, 1)[None])
        boxes, scores = decode_outputs(
            self.session.run(None, {self.input_name: tensor}), self.threshold
        )
        if not len(boxes):
            return []
        classes = scores.argmax(axis=1)
        confidence = scores.max(axis=1)
        wh = boxes.copy()
        wh[:, 2:] -= wh[:, :2]
        selected = []
        # Suppress only within a class; overlapping different objects remain available.
        for cls in np.unique(classes):
            indices = np.where(classes == cls)[0]
            keep = cv2.dnn.NMSBoxes(
                wh[indices].tolist(), confidence[indices].tolist(), self.threshold, 0.5
            )
            selected.extend(indices[np.asarray(keep, dtype=int).reshape(-1)].tolist())
        height, width = frame.shape[:2]
        results = []
        for index in sorted(selected, key=lambda i: float(confidence[i]), reverse=True)[:30]:
            x1, y1, x2, y2 = boxes[index]
            coords = [
                round(max(0, min(width, (x1 - left) / sx))),
                round(max(0, min(height, (y1 - top) / sy))),
                round(max(0, min(width, (x2 - left) / sx))),
                round(max(0, min(height, (y2 - top) / sy))),
            ]
            if coords[2] - coords[0] < 2 or coords[3] - coords[1] < 2:
                continue
            results.append(
                {
                    "label": COCO_LABELS[classes[index]],
                    "box_xyxy": coords,
                    "confidence": float(confidence[index]),
                    "source": "nanodet",
                }
            )
        return results


def merge_objects(detected, described):
    """Prefer fresh detector geometry for overlapping objects; keep novel VLM regions."""
    merged = list(detected)
    for item in described:
        a = item["box_xyxy"]
        duplicate = False
        for other in detected:
            b = other["box_xyxy"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
            if union > 0 and intersection / union >= 0.45:
                duplicate = True
                break
        if not duplicate:
            merged.append({**item, "source": "qwen"})
    return merged
