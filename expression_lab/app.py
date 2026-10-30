"""Run: uv run python app.py --setup, then uv run python app.py."""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
from contextlib import ExitStack
from pathlib import Path

import cv2
import onnxruntime as ort

from .configuration import DEFAULTS, PATH_OPTIONS, load_yaml, validate_options
from .dashboard import WINDOW, dashboard, save_snapshot
from .emotion import RAW_LABELS, EmotionRecognizer, download_detector, download_model
from .tracking import select_face


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    preliminary, _ = pre.parse_known_args(argv)
    parser = argparse.ArgumentParser(description="Detect faces and estimate expressions locally.")
    parser.add_argument("--config", type=Path, help="Strict YAML settings; CLI options override it")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        if name == "no_storage":
            parser.add_argument(flag, action="store_true", default=default)
        elif isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif name in PATH_OPTIONS:
            parser.add_argument(flag, type=Path, default=default)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, default=default)
    parser.add_argument(
        "--setup", action="store_true", help="Download and verify the pretrained CNN"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check models/runtime without opening a camera"
    )
    parser.add_argument(
        "--detector",
        choices=("yunet", "haar"),
        default="yunet",
        help="Face detector (default: yunet)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.25,
        help="Expression smoothing time in seconds (0 disables it)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("outputs/snapshots"),
        help="Directory used only when S is pressed",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    sources.add_argument("--video", type=Path, help="Read a local video file")
    sources.add_argument("--image", type=Path, help="Analyze a single local image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence [0,1]")
    parser.add_argument("--mirror", action="store_true", help="Mirror input horizontally")
    parser.add_argument("--headless", action="store_true", help="Process without opening a window")
    parser.add_argument(
        "--output", type=Path, help="Save dashboard (.jpg/.png for image, .mp4 for video)"
    )
    parser.add_argument(
        "--jsonl", type=Path, help="Save per-frame scores as newline-delimited JSON"
    )
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    if preliminary.config:
        try:
            allowed = {action.dest for action in parser._actions} - {"help", "config"}
            parser.set_defaults(**load_yaml(preliminary.config, allowed))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not math.isfinite(args.smoothing) or args.smoothing < 0:
        parser.error("--smoothing must be a finite nonnegative number")
    if args.setup and args.check:
        parser.error("--setup and --check are separate actions")
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.output:
        extensions = (".png", ".jpg", ".jpeg") if args.image else (".mp4",)
        if args.output.suffix.lower() not in extensions:
            parser.error(f"--output must use one of: {', '.join(extensions)}")
    if sum(bool(p) for p in (args.video, args.image, args.batch)) > 1:
        parser.error("Choose one source: image, video, or batch")
    inputs = {
        p.resolve()
        for p in (
            args.video,
            args.image,
            args.config,
            args.calibration,
            args.model_spec,
            args.auxiliary,
        )
        if p
    }
    export_paths = (
        args.output,
        args.jsonl,
        args.report,
        args.events_jsonl,
        args.csv,
        args.batch_output,
    )
    outputs = [p.resolve() for p in export_paths if p]
    if inputs.intersection(outputs) or len(outputs) != len(set(outputs)):
        parser.error("Input and output paths must be different")
    for path in export_paths:
        if path and path.exists():
            parser.error(f"Output already exists; choose a new filename: {path}")
    return args


def run(args):
    from .analytics import SessionAnalytics
    from .configuration import apply_calibration, recognizer_options
    from .events import EventDetector, WebhookDispatcher
    from .pipeline import FramePipeline, InferenceWorker

    if args.setup:
        if args.no_storage:
            raise ValueError("--no-storage forbids model downloads; use preinstalled models")
        print(f"CNN ready: {download_model()}")
        print(f"Detector ready: {download_detector()}")
        return 0
    if args.check:
        from .diagnostics import system_report

        report = system_report(args.detector)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    if args.batch:
        from .batch import process_batch

        return process_batch(args)
    for path in (args.video, args.image):
        if path and not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    if (
        not args.headless
        and sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        raise RuntimeError("No desktop display found. Use --headless to process without a window.")
    recognizer = EmotionRecognizer(args.threshold, args.detector, **recognizer_options(args))
    apply_calibration(recognizer, args)
    pipeline = FramePipeline(
        recognizer,
        0 if args.image else args.smoothing,
        args.min_overlap,
        args.detect_every,
        args.infer_every,
    )
    analytics = SessionAnalytics(args.history_size)
    print(
        "Local expression estimates only. Obtain consent; storage and webhooks require explicit options.",
        file=sys.stderr,
    )
    with ExitStack() as stack:
        if not args.headless:
            stack.callback(cv2.destroyAllWindows)
        capture, image = None, None
        if args.image:
            image = cv2.imread(str(args.image))
            if image is None:
                raise ValueError(f"Cannot decode image: {args.image}")
            source_fps = 1.0
        else:
            capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
            stack.callback(capture.release)
            if not capture.isOpened():
                raise RuntimeError(
                    "Cannot open input. Check the video path or webcam index/permissions."
                )
            if not args.video:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = 25.0
        log = _open_export(stack, args.jsonl)
        event_log = _open_export(stack, args.events_jsonl)
        csv_log = _open_export(stack, args.csv)
        csv_writer = None
        if csv_log:
            import csv

            csv_writer = csv.DictWriter(
                csv_log,
                fieldnames=[
                    "frame",
                    "timestamp_seconds",
                    "track_id",
                    "label",
                    "confidence",
                    "quality_accepted",
                    *RAW_LABELS,
                ],
            )
            csv_writer.writeheader()
        dispatcher = WebhookDispatcher(args.webhook) if args.webhook else None
        if dispatcher:
            stack.callback(dispatcher.close)
        events = EventDetector(
            args.event_label,
            args.event_seconds,
            args.event_confidence,
            args.event_cooldown,
            dispatcher.submit if dispatcher else None,
        )
        worker = InferenceWorker(pipeline) if args.async_pipeline else None
        if worker:
            stack.callback(worker.close)
        writer = None
        frames, face_count = 0, 0
        start = time.perf_counter()
        paused, multi = bool(args.image), False
        selected_id = None
        fps = 0.0
        frame, faces = None, []
        prefetched = None
        eof = False
        source_name = "IMAGE" if args.image else "VIDEO" if args.video else "WEBCAM"
        source_title = f"{source_name} / {args.detector.upper()}"
        while True:
            loop_start = time.perf_counter()
            new_frame = not paused or frames == 0
            if new_frame:
                if prefetched is not None:
                    assert worker is not None
                    frame, timestamp = prefetched
                    faces = worker.result()
                    prefetched = None
                else:
                    if eof:
                        break
                    if image is not None:
                        frame = image
                    else:
                        assert capture is not None
                        ok, frame = capture.read()
                        if not ok:
                            if args.video and frames:
                                break
                            raise RuntimeError(
                                "Input returned no frame. Check the camera or video decoder."
                            )
                    if args.mirror:
                        frame = cv2.flip(frame, 1)
                    timestamp = (
                        frames / source_fps
                        if args.video or args.image
                        else time.perf_counter() - start
                    )
                    if worker:
                        worker.submit(frame, timestamp)
                        faces = worker.result()
                    else:
                        faces = pipeline.process(frame, timestamp)
                selected_id = select_face(faces, selected_id)
                frames += 1
                face_count += len(faces)
                emitted = events.update(faces, timestamp)
                analytics.update(faces, timestamp, emitted)
                record = {
                    "schema_version": 3,
                    "frame": frames,
                    "timestamp_seconds": timestamp,
                    "frame_size_wh": [frame.shape[1], frame.shape[0]],
                    "detector": args.detector,
                    "engine": args.engine,
                    "smoothing_seconds": pipeline.tracker.smoothing,
                    "temperature": recognizer.temperature,
                    "detection_ran": pipeline.last_detection,
                    "inference_ran": pipeline.last_inference,
                    "events": emitted,
                    "auxiliary_available": recognizer.auxiliary is not None,
                    "consent_acknowledged": args.consent_ack,
                    "faces": [face.as_dict() for face in faces],
                }
                # At most one future frame: worker inference overlaps rendering/export of this frame.
                if worker and image is None and not (args.max_frames and frames >= args.max_frames):
                    assert capture is not None
                    ok, upcoming = capture.read()
                    if ok:
                        if args.mirror:
                            upcoming = cv2.flip(upcoming, 1)
                        next_time = (
                            frames / source_fps if args.video else time.perf_counter() - start
                        )
                        worker.submit(upcoming, next_time)
                        prefetched = (upcoming, next_time)
                    else:
                        eof = True
                rendered = dashboard(
                    frame,
                    faces,
                    fps,
                    selected_id,
                    False,
                    source_title,
                    privacy_blur=args.privacy_blur,
                    overlays=args.overlays,
                    multi=multi,
                    histories=analytics.tracks,
                )
                if log:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                if event_log:
                    for event in emitted:
                        event_log.write(json.dumps(event) + "\n")
                if csv_writer:
                    for face in faces:
                        csv_writer.writerow(
                            {
                                "frame": frames,
                                "timestamp_seconds": timestamp,
                                "track_id": face.track_id,
                                "label": face.label,
                                "confidence": face.confidence,
                                "quality_accepted": face.quality["accepted"]
                                if face.quality
                                else "",
                                **face.probabilities,
                            }
                        )
                if args.image:
                    print(json.dumps(record, indent=2, allow_nan=False))
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    if image is None:
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(args.output),
                                cv2.VideoWriter.fourcc(*"mp4v"),
                                source_fps,
                                (rendered.shape[1], rendered.shape[0]),
                            )
                            stack.callback(writer.release)
                            if not writer.isOpened():
                                raise RuntimeError(f"Cannot create output video: {args.output}")
                        writer.write(rendered)
                    elif not cv2.imwrite(str(args.output), rendered):
                        raise RuntimeError(f"Failed to write image: {args.output}")
                current_fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                fps = current_fps if not fps else 0.2 * current_fps + 0.8 * fps
            if not args.headless:
                assert frame is not None
                shown = dashboard(
                    frame,
                    faces,
                    fps,
                    selected_id,
                    paused,
                    source_title,
                    privacy_blur=args.privacy_blur,
                    overlays=args.overlays,
                    multi=multi,
                    histories=analytics.tracks,
                )
                cv2.imshow(WINDOW, shown)
                elapsed_ms = (time.perf_counter() - loop_start) * 1000
                delay = (
                    30
                    if paused
                    else max(1, round(1000 / source_fps - elapsed_ms))
                    if args.video
                    else 1
                )
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == 32 and image is None:
                    paused = not paused
                elif key == 9:
                    selected_id = select_face(faces, selected_id, cycle=True)
                elif key in (ord("m"), ord("M")):
                    multi = not multi
                elif key in (ord("s"), ord("S")):
                    if args.no_storage:
                        print("Snapshot disabled by --no-storage", file=sys.stderr)
                    else:
                        try:
                            print(
                                f"Snapshot saved: {save_snapshot(args.snapshot_dir, shown)}",
                                file=sys.stderr,
                            )
                        except (OSError, RuntimeError, cv2.error) as exc:
                            print(f"Snapshot failed: {exc}", file=sys.stderr)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if (args.headless and image is not None) or (
                args.max_frames and frames >= args.max_frames
            ):
                break
        if args.report:
            analytics.write_report(args.report)
        print(
            f"Processed {frames} frame(s), {face_count} face detection(s) in {time.perf_counter() - start:.2f}s.",
            file=sys.stderr,
        )
    return 0


def _open_export(stack, path):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return stack.enter_context(path.open("x", encoding="utf-8", newline=""))


def main(argv=None):
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        RuntimeError,
        cv2.error,
        urllib.error.URLError,
        ort.capi.onnxruntime_pybind11_state.Fail,
        ort.capi.onnxruntime_pybind11_state.InvalidGraph,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
