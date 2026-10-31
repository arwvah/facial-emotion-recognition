"""Read-only runtime/model diagnostics. Never open a capture device."""

import os
import platform

import cv2
import onnxruntime as ort

from .emotion import DETECTOR_PATH, DETECTOR_SHA256, MODEL_PATH, MODEL_SHA256, checksum


def system_report(detector="yunet", expression_model="ferplus", object_detector="none") -> dict:
    from .recognition_models import HSEMOTION_PATH, HSEMOTION_SHA256, NANODET_PATH, NANODET_SHA256

    models = {}
    for name, path, digest in (
        ("ferplus", MODEL_PATH, MODEL_SHA256),
        ("yunet", DETECTOR_PATH, DETECTOR_SHA256),
        ("hsemotion", HSEMOTION_PATH, HSEMOTION_SHA256),
        ("nanodet", NANODET_PATH, NANODET_SHA256),
    ):
        exists = path.is_file()
        models[name] = {
            "path": str(path),
            "exists": exists,
            "verified": exists and checksum(path) == digest,
        }
    ready = models[expression_model]["verified"] and (
        detector == "haar" or models["yunet"]["verified"]
    )
    ready = ready and (object_detector == "none" or models[object_detector]["verified"])
    return {
        "ready": ready,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "detector": detector,
        "expression_model": expression_model,
        "object_detector": object_detector,
        "models": models,
        "display_available": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if platform.system() == "Linux"
        else True,
        "camera_opened": False,
    }
