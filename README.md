# Expression Lab

A local Python application for eight-class facial-expression estimates: **happy, sad, angry,
neutral, surprise, disgust, fear, and contempt**. It detects faces in webcams, videos, or images
with YuNet and classifies their visible expressions. The standard launcher uses FER+.
The **1.3 recognition preset** adds:

- **NanoDet Plus:** fresh named boxes for 80 common object categories on every processed frame.
- **HSEmotion:** a 224×224 RGB CNN for all eight expressions, with augmentation and ambiguity rejection.
- **Qwen3-VL in LM Studio:** broader descriptions and approximate boxes for other objects.
- **R — Inspect:** select a smaller region for a closer look; boxes map back to the full frame.
- Face-quality guidance for small, dark, overexposed or blurry faces.

No model recognizes every object reliably. Small, unusual, occluded objects and subtle expressions
can still be missed or mislabeled. These additions have not been evaluated on your camera or a
representative labeled dataset. [Model details and limitations](MODEL_CARD.md).

```bash
./run.sh --setup-recognition    # Explicitly download and verify ~20 MB of additional models
# Load Qwen3-VL and start LM Studio's local server, then:
./run-vision.sh --vision-check
./run-vision.sh --no-storage
# Object boxes + expressions without LM Studio:
./run.sh --object-detector nanodet --expression-model hsemotion --no-align --tta \
  --expression-margin 0.08 --face-detector-size 960 --no-storage
```

Press **A** to ask, **R** to inspect a region, **T** to change the question, **V** for the vision
panel, and **M** for facial-expression histories. [Local vision setup and usage](VISION.md).

Expression estimates describe visible patterns, not someone's feelings. Obtain consent before
processing people. [Feature status](FEATURES.md) · [Privacy](PRIVACY.md) · [Model card](MODEL_CARD.md)
· [Development/evaluation](DEVELOPMENT.md) · [Verification](VERIFICATION.md).

## Install and start

Python 3.12 or 3.13. From this checkout, using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arwvah/facial-emotion-recognition.git
cd facial-emotion-recognition
uv sync --locked --extra api --extra optimize
./run.sh --setup
./run.sh --check
./run.sh --consent-ack
```

API/OpenVINO dependencies are optional. `uv sync --locked` installs only the core application
and development tools. The launcher preserves already installed optional extras.

For installation into your own virtual environment:

```bash
pip install .                 # Core desktop/CLI
pip install '.[api,optimize]'  # Add REST/WebSocket, OpenVINO and INT8 tooling
expression-lab --setup
expression-lab --check
expression-lab --consent-ack
```

Setup explicitly downloads and verifies the expression CNN (35 MB) and YuNet detector (233 KB).
Source checkouts use `models/`; installed wheels use `~/.cache/expression-lab/models/`.
Set `EXPRESSION_LAB_HOME=/path/to/data` to choose a different data root. Inference does not
automatically download weights. `--check` never opens the camera.

## Desktop and CLI

| Key | Action |
|---|---|
| Space | Pause/resume |
| Tab | Select next visible track |
| M | Switch eight-score view / multi-face cards and recent histories |
| S | Save a dashboard snapshot, unless `--no-storage` is active |
| Q / Esc | Quit |

```bash
./run.sh --video /path/to/clip.mp4
./run.sh --image /path/to/image.jpg --headless --output outputs/image.png
./run.sh --video /path/to/clip.mp4 --headless --tta \
  --output outputs/annotated.mp4 --jsonl outputs/frames.jsonl \
  --csv outputs/faces.csv --report outputs/session.html
./run.sh --video /path/to/clip.mp4 --headless --report outputs/session.json
./run.sh --camera 1 --mirror --threshold 0.65 --smoothing 0.4
./run.sh --config config.example.yaml --video /path/to/clip.mp4
```

CLI flags override YAML. YAML paths are relative to the configuration file. Unknown settings,
invalid values, input/output collisions, and existing export files are rejected.

Reports contain per-track dominant expressions, counts, transitions, and SVG timelines/charts.
They are standalone HTML or JSON, without external scripts, face images, or remote chart services.
History is bounded by `--history-size` (default 240 samples per track); counts cover the entire
session's fresh classification passes. Temporary track IDs do not identify people across gaps.

## Accuracy and robustness controls

- **Uncertain:** scores below `--threshold` return `uncertain`. Quality rejection also forces it.
- **Alignment:** YuNet eye landmarks rotate and consistently position crops before classification.
  Use `--no-align` to compare unaligned crops. Haar has no landmarks and uses a plain crop.
- **Quality:** `--min-face-size`, `--min-brightness`, `--max-brightness`, and `--min-sharpness`
  control rejection. `--quality-mode skip` avoids CNN work for rejected faces; `flag` retains
  model scores but still returns `uncertain`. Skipped classifications use a uniform fallback,
  explicitly marked `classification_skipped` and `uncertainty_reason: image_quality`.
- **Augmentation:** `--tta` averages original, flipped, and slightly cropped views, requiring
  three forward passes. It is optional; accuracy gains require evaluation.
- **Temperature:** `--temperature 1.5` is a manual adjustment. A fitted `--calibration file.json`
  must match the exact model/preprocessing contract. No trained calibration is bundled.

All eight probability scores retain their probability mass. `raw_probabilities` means the
current frame's scores after augmentation/temperature, before temporal smoothing. Schema 3
JSONL adds quality measurements/reasons, landmarks, alignment, auxiliary outputs, and whether
detection/classification passes ran. Reused results have `fresh_prediction: false`.

## Performance and model selection

```bash
./run.sh --video /path/to/clip.mp4 --async-pipeline --detect-every 3 --infer-every 2
./run.sh --video /path/to/clip.mp4 --engine opencv
./run.sh --video /path/to/clip.mp4 --engine openvino
./run.sh --model-spec model.ferplus.json --video /path/to/clip.mp4
```

Optical flow propagates boxes/landmarks between detector passes, rejecting failed motion or
blank/different crops. Inference can run in one worker thread with at most one prefetched frame,
overlapping rendering/export; video frames are preserved. In async mode, pause can leave one
already-read frame pending; it is not exported until processing resumes. Skipping can miss brief
changes, so compare it on representative videos. Default intervals are 1 (no skipping).

Adapters support ONNX Runtime, OpenCV DNN, and OpenVINO CPU. Other pretrained CNNs require an
explicit JSON manifest with checksum, class order, dimensions, color, and normalization;
inputs must be float32 NCHW. `model.ferplus.json` is a runnable example for the existing model.
An available ONNX Runtime accelerator can be selected with `--provider CUDAExecutionProvider`.
This needs a separately installed provider-enabled runtime and compatible hardware; unavailable
providers fail explicitly. The default installation and lockfile use the CPU runtime.

## Batch processing

```bash
./run.sh --batch /path/to/media --batch-output outputs/batch.json
./run.sh --batch /path/to/media --batch-output outputs/batch.csv --max-frames 300
./run.sh --batch /path/to/media --no-storage  # JSON to stdout, no export file
```

Recursively processes supported image/video files with one model load, a fresh tracker per file,
and per-file errors. Output includes status, frame/track counts, dominant expression, and counts.
A failed file produces an error row and nonzero final exit status; other files continue.

## Local REST and WebSocket API

```bash
uv run --extra api expression-lab-api --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: image/jpeg' --data-binary @/path/to/photo.jpg
```

`GET /health`, `GET /capabilities`, `POST /analyze`, and `WS /ws` are available.
Interactive API documentation is at `/docs`. Send encoded JPEG/PNG bytes for each WebSocket
message; responses include predictions and a bounded per-connection session timeline.
The API never opens the server webcam or writes uploaded frames/results to disk.

```python
import json
from pathlib import Path
from websockets.sync.client import connect

with connect("ws://127.0.0.1:8000/ws") as stream:
    stream.send(Path("photo.jpg").read_bytes())
    result = json.loads(stream.recv())
```

Set `EXPRESSION_LAB_API_TOKEN` to require `Authorization: Bearer ...` on inference endpoints
(including WebSocket upgrade headers). Remote binding additionally requires `--allow-remote`.
No permissive CORS is enabled. Default binding is loopback. The service is intended for local
experiments; use an authenticated HTTPS reverse proxy and deployment limits for remote hosting.

## Events and privacy exports

```bash
./run.sh --video /path/to/clip.mp4 --event-label happy --event-seconds 2 \
  --event-confidence 0.75 --event-cooldown 10 --events-jsonl outputs/events.jsonl
./run.sh --video /path/to/clip.mp4 --webhook http://127.0.0.1:9000/hook --no-local-only
./run.sh --video /path/to/clip.mp4 --privacy-blur --output outputs/pixelated.mp4
./run.sh --video /path/to/clip.mp4 --no-overlays --output outputs/plain.mp4
./run.sh --camera 0 --consent-ack --no-storage
```

Events require a sustained confident expression on the same visible track; gaps, quality
rejection, and label changes reset the timer. Webhooks send event metadata only, use a bounded
queue/timeout, and do not follow redirects. Delivery is opt-in; the API does not deliver webhooks.
`--no-storage` forbids exports, reports, downloads, and snapshots. `--no-overlays` exports input
resolution rather than the 1200×720 dashboard. Pixelation covers detected faces only and cannot
guarantee anonymization if a detector misses a face. See [PRIVACY.md](PRIVACY.md).

## AU, valence/arousal, evaluation and deployment

A supplied trained ONNX auxiliary model can output continuous valence/arousal and FACS action
units using `--auxiliary manifest.json`. These values appear in JSON and the selected-face
view. The adapter is tested, but **no AU/VA trained weights are bundled**. Without one, values
remain `null`; they are never fabricated from the expression label. `auxiliary.example.json`
is a contract template, not a downloadable or trained model.

See [DEVELOPMENT.md](DEVELOPMENT.md) for calibration, accuracy/FPS comparisons, group evaluation,
INT8 gating, OpenVINO export, Docker commands, CI, packaging, and auxiliary model contracts.
We have no labeled validation/test dataset, so there is **no claimed real-world accuracy,
calibration gain, fairness result, GPU speedup, or approved INT8 replacement**.
