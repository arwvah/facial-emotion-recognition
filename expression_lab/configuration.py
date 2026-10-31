"""Strict YAML configuration and shared CLI/API option validation."""

import json
import math
from pathlib import Path

import yaml

from .labels import RAW_LABELS
from .vision import DEFAULT_PROMPT, client_from_args, validate_prompt

DEFAULTS = {
    "vision": False,
    "vision_check": False,
    "vision_url": "http://127.0.0.1:1234/v1",
    "vision_model": "qwen3-vl-2b-instruct",
    "vision_prompt": DEFAULT_PROMPT,
    "vision_interval": 0.0,
    "vision_timeout": 60.0,
    "vision_max_tokens": 512,
    "vision_boxes": True,
    "vision_max_side": 768,
    "vision_jsonl": None,
    "temperature": 1.0,
    "align": True,
    "tta": False,
    "min_face_size": 48,
    "min_brightness": 25.0,
    "max_brightness": 240.0,
    "min_sharpness": 20.0,
    "quality_mode": "skip",
    "engine": "onnx",
    "provider": "CPUExecutionProvider",
    "model_spec": None,
    "auxiliary": None,
    "calibration": None,
    "detect_every": 1,
    "infer_every": 1,
    "async_pipeline": False,
    "min_overlap": 0.3,
    "history_size": 240,
    "privacy_blur": False,
    "overlays": True,
    "no_storage": False,
    "local_only": True,
    "consent_ack": False,
    "report": None,
    "events_jsonl": None,
    "webhook": None,
    "event_label": "happy",
    "event_seconds": 2.0,
    "event_confidence": 0.7,
    "event_cooldown": 10.0,
    "batch": None,
    "batch_output": None,
    "csv": None,
}
PATH_OPTIONS = {
    "vision_jsonl",
    "video",
    "image",
    "output",
    "jsonl",
    "snapshot_dir",
    "model_spec",
    "auxiliary",
    "calibration",
    "report",
    "events_jsonl",
    "batch",
    "batch_output",
    "csv",
}
BOOL_OPTIONS = {
    "vision_boxes",
    "vision",
    "vision_check",
    "align",
    "tta",
    "async_pipeline",
    "privacy_blur",
    "overlays",
    "no_storage",
    "local_only",
    "consent_ack",
    "headless",
    "mirror",
}


def load_yaml(path: Path, allowed: set[str]) -> dict:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {exc}") from exc
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError(
            f"Config must be a mapping with known keys; unknown: {set(data) - allowed if isinstance(data, dict) else 'non-mapping'}"
        )
    for name, value in data.items():
        if name in BOOL_OPTIONS and not isinstance(value, bool):
            raise ValueError(f"{name} must be a YAML boolean")
        if name in PATH_OPTIONS and value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a path string")
            data[name] = (path.parent / value).resolve()
    return data


def validate_options(args):
    client_from_args(args)  # Validates settings without connecting to LM Studio.
    validate_prompt(args.vision_prompt)
    if (
        isinstance(args.vision_interval, bool)
        or not isinstance(args.vision_interval, (int, float))
        or not math.isfinite(args.vision_interval)
        or args.vision_interval < 0
    ):
        raise ValueError("vision-interval must be a finite nonnegative number of seconds")
    if args.vision_jsonl and not args.vision:
        raise ValueError("--vision-jsonl requires --vision")
    if args.vision and args.batch:
        raise ValueError("Vision supports image/video/camera input; batch mode is expression-only")
    if isinstance(args.camera, bool) or not isinstance(args.camera, int) or args.camera < 0:
        raise ValueError("camera must be a nonnegative integer")
    if args.max_frames is not None and (
        isinstance(args.max_frames, bool)
        or not isinstance(args.max_frames, int)
        or args.max_frames < 1
    ):
        raise ValueError("max_frames must be a positive integer")
    for name in (
        "threshold",
        "temperature",
        "smoothing",
        "min_brightness",
        "max_brightness",
        "min_sharpness",
        "min_overlap",
        "event_seconds",
        "event_confidence",
        "event_cooldown",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    for name in ("detect_every", "infer_every", "min_face_size", "history_size"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= args.threshold <= 1 or args.temperature <= 0 or args.smoothing < 0:
        raise ValueError("Invalid threshold, temperature, or smoothing")
    if not 0 < args.min_overlap <= 1 or not 0 <= args.event_confidence <= 1:
        raise ValueError("Invalid overlap or event confidence")
    if not 0 <= args.min_brightness < args.max_brightness <= 255 or args.min_sharpness < 0:
        raise ValueError("Invalid quality thresholds")
    if args.event_seconds < 0 or args.event_cooldown < 0:
        raise ValueError("Event times must be nonnegative")
    if args.engine not in ("onnx", "opencv", "openvino") or args.detector not in ("yunet", "haar"):
        raise ValueError("Unknown inference engine or detector")
    if args.quality_mode not in ("skip", "flag") or args.event_label not in RAW_LABELS:
        raise ValueError("Unknown quality mode or event label")
    if args.calibration and args.temperature != 1:
        raise ValueError("Use either --calibration or --temperature")
    if args.no_storage and any(
        getattr(args, key)
        for key in (
            "output",
            "jsonl",
            "report",
            "events_jsonl",
            "csv",
            "batch_output",
            "vision_jsonl",
        )
    ):
        raise ValueError("--no-storage forbids file export/report flags")
    if args.webhook and args.local_only:
        raise ValueError("Webhook delivery requires --no-local-only explicitly")
    if args.report and args.report.suffix.lower() not in (".html", ".json"):
        raise ValueError("--report must end in .html or .json")
    if args.batch_output and args.batch_output.suffix.lower() not in (".csv", ".json"):
        raise ValueError("--batch-output must end in .csv or .json")
    if args.batch and not (args.batch_output or args.no_storage):
        raise ValueError("Batch mode requires --batch-output or --no-storage (stdout only)")


def recognizer_options(args) -> dict:
    from .quality import QualityPolicy

    return {
        "engine": args.engine,
        "provider": args.provider,
        "model_spec": args.model_spec,
        "temperature": args.temperature,
        "align": args.align,
        "tta": args.tta,
        "quality": QualityPolicy(
            args.min_face_size, args.min_brightness, args.max_brightness, args.min_sharpness
        ),
        "quality_mode": args.quality_mode,
        "auxiliary": args.auxiliary,
    }


def apply_calibration(recognizer, args):
    if args.calibration:
        data = json.loads(args.calibration.read_text())
        expected = {"align": args.align, "tta": args.tta}
        if (
            data.get("model_sha256") != recognizer.backend.spec.sha256
            or data.get("model_contract") != recognizer.backend.spec.signature()
            or data.get("preprocessing") != expected
        ):
            raise ValueError(
                "Calibration belongs to a different model or alignment/TTA configuration"
            )
        temperature = float(data["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid calibration temperature")
        recognizer.temperature = temperature
