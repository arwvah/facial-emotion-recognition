# Local vision with LM Studio

Ask about objects, scenes and visible text using the full camera frame. Expression Lab keeps
a separate expression CNN and runs the vision-language model separately. Answers can be
incomplete or wrong; this is not exhaustive object detection, identity recognition or emotion
measurement. The application never runs instructions returned by the model.

## Prepare LM Studio

Download **lmstudio-community/Qwen3-VL-2B-Instruct-GGUF**, **Q4_K_M**, including its matching
vision projector (`mmproj`). Load it and start the local server in the Developer tab, listening
on **127.0.0.1:1234**. The model ID used by this project is `qwen3-vl-2b-instruct`.
If you assigned a different ID, pass `--vision-model YOUR-ID`.

The equivalent CLI setup, when the model is not already loaded:

```bash
lms ls
lms load qwen3-vl-2b-instruct --context-length 4096 --parallel 1
lms server start --port 1234 --bind 127.0.0.1
./run-vision.sh --vision-check
```

Keep an already working loaded model; don't load a duplicate copy. LM Studio chooses GPU
placement automatically unless you override it. Memory usage also includes the vision encoder,
context and compute buffers. If loading fails, reduce context/parallelism or GPU offload in LM
Studio; close unused model instances. The application does not change your runtime load settings.
`--vision-check` verifies connectivity and the configured model ID without opening a camera;
image inference is the definitive vision-support check.

References: [model](https://huggingface.co/lmstudio-community/Qwen3-VL-2B-Instruct-GGUF),
[LM Studio server](https://lmstudio.ai/docs/developer/core/server),
[load settings](https://lmstudio.ai/docs/cli/local-models/load).

## Camera, images and video

Run `./run.sh --setup-recognition` once before using the enhanced launcher. It downloads
checksum-pinned NanoDet Plus (~3.8 MB), HSEmotion (~16 MB), and YuNet if missing. No weights are
bundled or downloaded during inference. `./run-vision.sh --check` verifies the selected assets.

The launcher enables a two-part object path: NanoDet detects 80 COCO categories on every
processed frame; Qwen describes the scene and locates up to six additional regions. Fresh
NanoDet geometry wins when boxes overlap substantially. Other Qwen regions remain visible and
use the existing short-term optical-flow tracking. Neither path guarantees exhaustive recognition.
Names appear below cyan rectangles, or inside the bottom edge when space is tight.

The launcher and `vision.example.yaml` also select HSEmotion with RGB224 preprocessing,
three-view augmentation, an 0.08 top-two score margin, and a 960-pixel face-detection limit.
HSEmotion uses unaligned bounding crops to match the author's preprocessing. Plain `./run.sh`
retains FER+ and its existing defaults. These presets are starting points, not calibrated settings.

Qwen samples every two seconds when idle. Use `--vision-interval 0` for manual requests.
`--no-vision-boxes` disables Qwen boxes; `--object-detector none` disables NanoDet boxes.
`--no-overlays` hides all annotations. Both object paths work independently of the vision panel.

```bash
./run-vision.sh --no-storage
./run.sh --config vision.example.yaml
./run-vision.sh --image photo.jpg --headless --no-storage \
  --vision-prompt 'What objects are visible?'
./run-vision.sh --image sign.jpg --headless --no-storage \
  --vision-prompt 'Transcribe the visible text.'
./run-vision.sh --video clip.mp4 --vision-interval 3
./run-vision.sh --video clip.mp4 --headless --vision-interval 3 \
  --vision-jsonl outputs/descriptions.jsonl
```

| Key | Action |
|---|---|
| A | Capture and ask about the current frame |
| R | Inspect a region: drag a box, Enter/Space confirms, C cancels; available when idle |
| T | Type a new question; Enter submits, Escape cancels |
| V | Toggle the local vision panel |
| M | Show the existing multi-face panel |
| Space / Tab / S / Q | Existing pause, select face, snapshot and quit controls |

For small electronics, tools, labels or unusual objects, press R and draw tightly around the
object. This sends the original-resolution crop to Qwen without the surrounding clutter.
Results carry `region_xywh` and coordinates translated into the original full frame. Inspection
is a one-shot request; automatic full-frame sampling resumes afterward. It improves the input
view but cannot add knowledge the model lacks.

Typing uses printable ASCII in the OpenCV window. CLI/API questions and JSON answers support
Unicode; the desktop font replaces unsupported characters. Long answers are shortened on screen;
stdout and optional JSONL preserve the complete response. T is available when no request is busy.

Images are analyzed once automatically. Interactive image exports should use `--headless` when
the saved dashboard must include the answer; interactive snapshots use S after the answer arrives.
Positive `--vision-interval` enables sampling based on
source timestamps, with at most one request in flight. Busy intervals are skipped, not queued;
fast offline video processing does not guarantee every interval gets analyzed. Image-headless mode
waits for its answer. Video completion/quit waits for the final in-flight request (bounded by the
configured timeout), and writes/prints that result even after the last frame.

The panel and JSON results identify the **source frame and timestamp**, since the answer may
refer to an older frame than the current display. `--jsonl` includes the latest completed answer
as an optional `vision` field; use `--vision-jsonl` for one record per completed request, including
final results and errors. Vision text is not merged into expression counts or session reports.
Batch folder mode remains expression-only and rejects `--vision` rather than silently ignoring it.

Object overlays replay recent optical flow when a delayed response arrives, then follow motion
between model calls. They disappear after tracking failure, a frame-size change, lost frame
history, or five seconds of source time. The grayscale history is bounded to 180 frames with
longest side 384 pixels. Fast/large motion, occlusion, low texture and zoom can lose boxes;
press A again or wait for the next sample. Turning off overlays removes all rectangles/names.

Results include `objects: [{label, box_xyxy}]` in **original source-frame pixels**,
`frame_size_wh`, `object_coordinates`, and `objects_rejected`. Invalid/reversed/out-of-range boxes
are rejected rather than guessed. Per-frame JSONL also has `object_overlays` for the coordinates
actually displayed on that current frame. Model source boxes remain unchanged. Names do not
identify a person; Qwen confidence percentages are not invented. NanoDet results include its
raw confidence score and `source: nanodet`; retained Qwen overlays have `source: qwen`.
Detector scores are not calibrated probabilities of correctness. A truncated structured response fails
explicitly instead of drawing partial boxes.

## API

```bash
uv run --extra api expression-lab-api --config vision.example.yaml
curl -X POST 'http://127.0.0.1:8000/describe?question=What%20is%20visible%3F' \
  -H 'Content-Type: image/jpeg' --data-binary @photo.jpg
```

`POST /describe` accepts raw JPEG/PNG/WebP bytes like `/analyze`; `question` is an optional query
parameter. Requests are capped at 8 MiB and four megapixels, questions at 2048 characters. This
route is disabled unless `vision: true`. It uses the same API bearer-token/origin checks, returns
429 while a vision request is busy, 422 for invalid input, and 503 if LM Studio is unavailable.
`/analyze` and `/ws` return expression results plus `objects` from NanoDet when enabled.
`/describe` returns Qwen results; the API does not fuse these separate requests. `/capabilities` reports vision configuration.
When face privacy blur is disabled, `/describe` does not load or require the expression CNN.

The LM Studio endpoint must use the literal loopback IP `127.0.0.1` or `[::1]` and an explicit
port followed by `/v1`. Remote URLs, credentials in URLs, redirects and environment proxies are
rejected/bypassed. If LM Studio authentication is enabled, set `LM_STUDIO_API_TOKEN` in the
process environment; never put the token in YAML or Git. This is separate from
`EXPRESSION_LAB_API_TOKEN`, which protects Expression Lab's API.

## Tuning and privacy

Core defaults: `--vision-max-side 768` (1024 in the enhanced launcher), `--vision-max-tokens 512`, `--vision-timeout 60`,
`--vision-boxes` enabled. The vision launcher uses `--vision-interval 2`; the core CLI default is 0.
Increase image size for small text at the cost of latency/memory. Text-only responses
with `finish_reason: length` were cut off at the generation limit. Model answers have no invented
confidence percentage. Request errors are visible and headless failures return nonzero.

No images are saved by the vision client. `--no-storage` rejects vision exports too; stdout can
still be redirected by the caller. `--privacy-blur` pixelates detected faces before sending the
frame to LM Studio. Missed detections and surrounding appearance can still identify people.
The LM Studio process can have its own request logs, so configure it separately if retention
matters. No cloud service, external model download, tool call, or webcam access is triggered by
merely importing or starting the API. Camera access begins only when the camera CLI is launched.
