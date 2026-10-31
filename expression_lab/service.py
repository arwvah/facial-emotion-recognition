"""Local FastAPI REST/WebSocket inference. Uploaded frames are never written to disk."""

import argparse
import asyncio
import os
import secrets
import threading
import time

import cv2
import numpy as np


def decode_frame(data: bytes):
    if not data or len(data) > 8 * 1024 * 1024:
        raise ValueError("Send an encoded image between 1 byte and 8 MiB")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[0] * frame.shape[1] > 4_000_000:
        raise ValueError("Invalid image, or image exceeds four megapixels")
    return frame


def create_app(config=None, recognizer_factory=None, vision_factory=None):
    try:
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install API support: pip install '.[api]'") from exc
    from .analytics import SessionAnalytics
    from .app import parse_args
    from .configuration import apply_calibration, recognizer_options
    from .emotion import EmotionRecognizer
    from .pipeline import FramePipeline
    from .vision import VisionBusy, client_from_args

    args = parse_args(
        (["--config", str(config)] if config else []) + ["--headless", "--no-storage"]
    )
    if args.webhook:
        raise ValueError("The API does not send webhooks; use the CLI event options instead")
    token = os.environ.get("EXPRESSION_LAB_API_TOKEN", "")
    model = None
    model_lock = threading.Lock()
    vision_client = (
        (vision_factory() if vision_factory else client_from_args(args)) if args.vision else None
    )
    app = FastAPI(
        title="Expression Lab local inference",
        version="1.1.0",
        description="Local expression estimates, not feelings. Obtain consent before submitting images. Frames are processed in memory; no image storage or webhooks.",
    )

    def authorized(headers):
        return not token or secrets.compare_digest(
            headers.get("authorization", ""), f"Bearer {token}"
        )

    def origin_allowed(headers):
        from urllib.parse import urlsplit

        origin = headers.get("origin")
        return not origin or urlsplit(origin).hostname in ("localhost", "127.0.0.1", "::1")

    def get_model():
        nonlocal model
        if model is None:
            model = (
                recognizer_factory()
                if recognizer_factory
                else EmotionRecognizer(args.threshold, args.detector, **recognizer_options(args))
            )
            if not recognizer_factory:
                apply_calibration(model, args)
        return model

    def process(data, pipeline=None):
        frame = decode_frame(data)
        with model_lock:
            recognizer = get_model()
            if pipeline is None:
                pipeline = FramePipeline(recognizer, 0, args.min_overlap)
            timestamp = time.monotonic()
            faces = pipeline.process(frame, timestamp)
        return {
            "schema_version": 3,
            "frame_size_wh": [frame.shape[1], frame.shape[0]],
            "timestamp_seconds": timestamp,
            "faces": [face.as_dict() for face in faces],
            "storage": "memory_only",
            "auxiliary_available": recognizer.auxiliary is not None,
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "storage": "memory_only",
            "camera_opened": False,
            "model_loaded": model is not None,
        }

    @app.get("/capabilities")
    def capabilities():
        from .labels import TARGET_LABELS

        return {
            "expressions": TARGET_LABELS,
            "quality_checks": True,
            "alignment": args.align,
            "tta": args.tta,
            "auxiliary_configured": bool(args.auxiliary),
            "vision_enabled": args.vision,
            "vision_model": args.vision_model if args.vision else None,
            "notice": "Visible expression estimates, not feelings. Obtain consent before submitting images.",
        }

    @app.post("/analyze")
    async def analyze(request: Request):
        if not authorized(request.headers) or not origin_allowed(request.headers):
            raise HTTPException(403, "Access denied")
        try:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 8 * 1024 * 1024:
                    raise HTTPException(413, "Image exceeds 8 MiB")
            return await asyncio.to_thread(process, bytes(data))
        except (ValueError, cv2.error) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, "Model unavailable; run --check and --setup locally") from exc

    @app.post("/describe")
    async def describe(request: Request):
        if not authorized(request.headers) or not origin_allowed(request.headers):
            raise HTTPException(403, "Access denied")
        if vision_client is None:
            raise HTTPException(503, "Vision disabled; enable vision in the server YAML config")
        try:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 8 * 1024 * 1024:
                    raise HTTPException(413, "Image exceeds 8 MiB")
            frame = decode_frame(bytes(data))
            if args.privacy_blur:
                from .dashboard import dashboard

                def blur():
                    with model_lock:
                        faces = get_model().detect(frame)
                    return dashboard(frame, faces, 0, privacy_blur=True, overlays=False)

                frame = await asyncio.to_thread(blur)
            prompt = request.query_params.get("question", args.vision_prompt)
            return await asyncio.to_thread(vision_client.describe, frame, prompt)
        except VisionBusy as exc:
            raise HTTPException(429, str(exc)) from exc
        except (ValueError, cv2.error) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                503, "Local vision unavailable; check LM Studio model and server"
            ) from exc

    @app.websocket("/ws")
    async def stream(socket: WebSocket):
        if not authorized(socket.headers) or not origin_allowed(socket.headers):
            await socket.close(code=1008)
            return
        await socket.accept()
        timeline = SessionAnalytics(args.history_size)
        try:
            with model_lock:
                recognizer = get_model()
                pipeline = FramePipeline(
                    recognizer,
                    args.smoothing,
                    args.min_overlap,
                    args.detect_every,
                    args.infer_every,
                )
            start = time.monotonic()
            while True:
                data = await socket.receive_bytes()
                result = await asyncio.to_thread(process, data, pipeline)
                timestamp = time.monotonic() - start
                faces = pipeline.previous_faces
                timeline.update(faces, timestamp)
                result["session"] = timeline.summary()
                await socket.send_json(result)
        except WebSocketDisconnect:
            pass
        except (ValueError, OSError, RuntimeError, cv2.error):
            await socket.send_json({"error": "Invalid frame or unavailable model"})
            await socket.close(code=1003)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1") and (
        not args.allow_remote or not os.environ.get("EXPRESSION_LAB_API_TOKEN")
    ):
        parser.error("Remote binding requires --allow-remote and EXPRESSION_LAB_API_TOKEN")
    try:
        import uvicorn

        app = create_app(args.config)
        uvicorn.run(
            app, host=args.host, port=args.port, access_log=False, ws_max_size=8 * 1024 * 1024
        )
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
