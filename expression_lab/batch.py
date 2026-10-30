"""Offline folder processing with one model load and one independent timeline per file."""

import csv
import json
from collections import Counter

import cv2

from .analytics import SessionAnalytics
from .configuration import apply_calibration, recognizer_options
from .emotion import EmotionRecognizer
from .pipeline import FramePipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def process_batch(args):
    if not args.batch.is_dir():
        raise ValueError("--batch must be a directory")
    if any((args.output, args.jsonl, args.report, args.csv, args.events_jsonl, args.webhook)):
        raise ValueError(
            "Batch exports use --batch-output; run individual files for video/session/event exports"
        )
    recognizer = EmotionRecognizer(args.threshold, args.detector, **recognizer_options(args))
    apply_calibration(recognizer, args)
    files = sorted(
        path
        for path in args.batch.rglob("*")
        if not path.is_symlink()
        and path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    )
    if not files:
        raise ValueError("No supported images/videos in the batch folder")
    results = []
    for path in files:
        analytics = SessionAnalytics(args.history_size)
        pipeline = FramePipeline(
            recognizer, args.smoothing, args.min_overlap, args.detect_every, args.infer_every
        )
        capture = None
        try:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError("Cannot decode image")
                frames = [(0.0, frame)]
            else:
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError("Cannot open video")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not fps or not 0 < fps < 1000:
                    fps = 25.0

                def iterate(source=capture, source_fps=fps):
                    index = 0
                    while True:
                        ok, frame = source.read()
                        if not ok:
                            break
                        yield index / source_fps, frame
                        index += 1
                        if args.max_frames and index >= args.max_frames:
                            break

                frames = iterate()
            for timestamp, frame in frames:
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                faces = pipeline.process(frame, timestamp)
                analytics.update(faces, timestamp)
            summary = analytics.summary()
            if not summary["frames"]:
                raise ValueError("No decodable frames")
            counts: Counter[str] = Counter()
            for track in summary["tracks"]:
                counts.update(track["label_counts"])
            dominant = max(counts, key=lambda key: counts[key]) if counts else "no_face"
            results.append(
                {
                    "file": str(path.relative_to(args.batch)),
                    "status": "ok",
                    "frames": summary["frames"],
                    "tracks": len(summary["tracks"]),
                    "dominant_expression": dominant,
                    "label_counts": dict(counts),
                    "error": None,
                }
            )
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            results.append(
                {
                    "file": str(path.relative_to(args.batch)),
                    "status": "error",
                    "frames": analytics.frames,
                    "tracks": len(analytics.tracks),
                    "dominant_expression": None,
                    "label_counts": {},
                    "error": str(exc),
                }
            )
        finally:
            if capture is not None:
                capture.release()
    if args.batch_output:
        args.batch_output.parent.mkdir(parents=True, exist_ok=True)
        with args.batch_output.open("x", encoding="utf-8", newline="") as handle:
            if args.batch_output.suffix.lower() == ".csv":
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(
                    {**row, "label_counts": json.dumps(row["label_counts"])} for row in results
                )
            else:
                json.dump(
                    {"schema_version": 1, "files": results}, handle, indent=2, allow_nan=False
                )
    else:
        print(json.dumps({"schema_version": 1, "files": results}, indent=2, allow_nan=False))
    return int(any(row["status"] == "error" for row in results))
