# Verification — 2026-10-31

## Current release: 1.1 — LM Studio vision

- Confirmed the downloaded LM Studio model is Qwen3-VL 2B Instruct Q4_K_M with vision support,
  about 1.93 GB including its projector. Started the API on `127.0.0.1:1234` and used the working
  local `qwen3-vl-2b-instruct` instance. Temporary duplicate test instances were unloaded. Initial
  duplicate GPU-load attempts failed on memory allocation; the existing working model was retained.
- Actual image questions passed through the new client, integrated headless CLI and `/describe`
  API: a generated card was described as a red circle and blue square, and a focused transcription
  request returned `LOCAL VISION 42`. A combined question omitted the text, demonstrating that
  answers are not exhaustive. No test result establishes general recognition accuracy.
- Visually inspected `outputs/vision-test-preview.png`. The panel shows the question, answer,
  source frame/time and controls. The original eight-class expression behavior remains available.
- Processed the actual 24-frame video fixture with sampled vision and retained all 36 face
  detections. Two completed descriptions identify source frames 1 and 23; the final in-flight
  response was printed and exported after video EOF. Sampling drops busy intervals by design.
- **72 tests passed**, including real loopback HTTP image transport, image resizing/color,
  proxy bypass, redirect/remote rejection, request bounds, model availability, malformed results,
  single-request concurrency, capture identity, no-storage, error exit codes, question typing,
  and API authentication/disabled/busy behavior. Tests require no LM Studio installation or weights.
- Ruff and formatting passed; mypy passed all **18 implementation modules**. The upstream
  Starlette/httpx test-client deprecation warning remains non-fatal.
- Built version 1.1 wheel/sdist and checked inclusion of the vision module, launcher, config and
  documentation. Installed the wheel into a clean temporary environment outside the checkout;
  actual LM Studio image inference passed under `--no-storage`, with no output files created.

Physical webcam use and broad object/OCR accuracy remain untested. LM Studio runtime logging is
outside the application's no-storage control. The vision model does not supply calibrated
confidence, object boxes, reliable identity, feelings, AU or VA outputs. See VISION.md.

## Release 1.0

- **61 tests passed**, including REST/WebSocket contracts, real synthetic ONNX execution,
  alignment, quality rejection, temperature fitting, data-split checks, analytics, events,
  configuration, tracking, async queue behavior, paired OpenVINO weight integrity and snapshot
  suppression in no-storage mode. Earlier release counts below are historical.
- Ruff lint and formatting passed; mypy passed across all **17 implementation modules**.
  FastAPI's test client emits an upstream Starlette/httpx deprecation warning; tests still pass.
- `uv lock --check`, `uv build`, and the launcher `--check` passed. Both production model hashes
  verify; the check command does not open the camera.
- Built wheel/sdist contents were checked: namespaced implementation and console entry points
  in the wheel; docs/config/examples/tests in the source distribution; no model weights or local
  test outputs bundled. Installed the wheel with optional API/optimization dependencies into a
  clean temporary virtual environment, outside the checkout. All four entry points launch;
  actual image inference returns one aligned face with null auxiliary fields and writes no
  output files under `--no-storage`.
- Real CPU inference passed with ONNX Runtime, OpenCV DNN and OpenVINO. Static INT8 conversion
  and paired XML/BIN OpenVINO export also produced runnable candidates. The INT8 validation
  artifact explicitly says `accepted: null`, `requires_held_out_evaluation`; the original CNN
  remains unchanged. One-image conversion smoke data does not establish accuracy preservation.
- Actual aligned/TTA image processing and HTML export passed. A 24-frame constructed two-face,
  one-face, blank, returning-face clip passed with async processing, detection every three frames,
  inference every two frames, and pixelated output. All 24 exported video frames decoded; JSONL
  face counts matched the expected sequence. There were 11 detection passes and 17 classification
  passes; these count scheduled stages, not individual model forward calls.
- Recursive batch processing passed on an image and video with both CSV and JSON summaries.
  Real HTTP and WebSocket requests passed through the production CNN on a temporary loopback
  server; that server was stopped afterward. Event delivery passed against a loopback HTTP
  receiver; no event was sent to an external service. API webhook configuration is rejected.
- Calibration, evaluation and artifact loading were exercised end to end on explicitly labeled
  **synthetic software fixtures**, including model/preprocessing binding. Their metrics are not
  real-world expression-recognition results.
- Visually inspected `outputs/report-preview.png`, `outputs/multi-face-v1.png`, and
  `outputs/auxiliary-layout-fixture.png`. Auxiliary layout values are synthetic fixtures; no
  trained AU/VA model was supplied. Local reports include `outputs/async-v1.html`.

- Published publicly at [arwvah/facial-emotion-recognition](https://github.com/arwvah/facial-emotion-recognition).
  The [initial GitHub Actions run](https://github.com/arwvah/facial-emotion-recognition/actions/runs/41130512220)
  passed dependency installation, lint, formatting, mypy and all 61 tests on the hosted Linux runner.
  Downloaded weights, local input/output images and videos, environments and build artifacts are
  excluded from Git.

**Unverified / data dependent:** physical webcam and GPU execution; Docker build/run (daemon
access denied, passwordless sudo unavailable); real accuracy, calibration improvement and demographic fairness; trained
AU/VA outputs; production INT8 approval. No representative labeled dataset or auxiliary weights
were provided. See FEATURES.md for the full implementation/status matrix.

## Original release

Verified locally with Python 3.12.14, OpenCV 4.14.0.94, ONNX Runtime 1.31.1,
and the dependency versions in `uv.lock`.

- `uv run python -m unittest discover -s tests -v`: **12 tests passed**.
- `uv run ruff check .`: **passed**.
- Downloaded CNN size and SHA-256 match the pinned original ONNX Git LFS metadata.
- Actual CNN inference on the [OpenCV reference image](https://github.com/opencv/opencv/blob/4.12.0/samples/data/lena.jpg):
  one face detected, valid eight-class scores summing to 1, image dashboard exported and
  visually inspected. This is a functional smoke test, without a ground-truth accuracy claim.
- Created a 12-frame, 12 FPS clip: six reference-image frames, followed by six blank frames.
  Headless processing produced exactly six face detections and none on the blank frames.
- Reopened and decoded the exported MP4: exactly 12 frames at 1200×720 and 12 FPS.
  JSONL contained 12 matching records; checked timestamps, probability sums, and box bounds.
- Desktop window mode processed three sample-video frames and exited successfully.
  On this Wayland desktop, OpenCV used its available XCB fallback.
- Missing-file input returned a concise error and nonzero exit status.

Local test artifacts are in ignored `outputs/`, including `preview.png`, `test-annotated.mp4`,
and the JSONL records. The upstream reference image is a local test input, not bundled source.

The physical webcam was not opened. No training, four-class accuracy benchmark, or live
webcam performance measurement was performed.

## Version 0.2 improvements — 2026-10-31

- Unit/regression suite: **28 tests passed**; Ruff checks and formatting passed.
- Both downloaded models match pinned upstream SHA-256 hashes. The FER+ model is unchanged;
  YuNet replaces Haar as the default detector. Actual inference also passed with the Haar option.
- Actual image inference/export passed with the new detector. Visually inspected the two-face
  dashboard in `outputs/two-faces-preview.png`, including track IDs and selected-face highlighting.
- Actual 24-frame, 12 FPS multi-face clip: eight frames with two faces, four with one face,
  four blank frames, then eight with two returning faces. Observed the expected detection
  counts on every frame; independent IDs stayed stable while present, absent tracks disappeared
  immediately, and returning faces received new IDs.
- Reopened `outputs/tracking-annotated.mp4`: all 24 frames decoded at 1200×720 and 12 FPS.
  Validated all JSONL timestamps, box bounds, detection scores, and raw/smoothed score sums.
- Simulated pause/resume at the application event loop: capture and inference advanced twice
  across four UI polls, with exactly two JSONL records. Separate tests cover face selection,
  reordered detections, new/distant tracks, time gaps, smoothing math, and no-smoothing mode.
- Updated desktop window mode processed three multi-face video frames and exited cleanly via XCB.
- Snapshot PNGs decode correctly; filename collisions fail without overwriting an existing file.
- Missing-display errors point to `--headless` before camera initialization. `--check` is tested
  to avoid camera access; the launcher runs from outside the project directory.

The test clip is constructed from a public OpenCV sample and its mirror; it verifies tracking
and export mechanics, not person identification or classification accuracy. The physical
webcam remains untested, and spatial IDs may switch when real faces cross or overlap.

## Version 0.3: all eight expressions — 2026-10-31

- Enabled surprise, disgust, fear, and contempt alongside happy, sad, angry, and neutral.
  Existing pretrained weights already output these classes; no weights or preprocessing changed.
- **29 tests passed**, including exact class-index mapping, low-confidence rejection for the
  added labels, their preservation through smoothing/export, and dashboard rendering for every label.
- Ruff lint/format checks passed. Visually inspected all eight dashboard rows in
  `outputs/eight-emotions-preview.png`.
- Actual two-face image inference/export passed. Processed and decoded the 24-frame integration
  video again, validating face transitions, all eight score keys, labels, and score sums.
  Artifacts: `outputs/eight-emotions-video.mp4` and `outputs/eight-emotions-video.jsonl`.
- Class-mapping tests use controlled scores; the real sample clip is not an accuracy benchmark
  for the four added expressions. Webcam hardware remains untested.
