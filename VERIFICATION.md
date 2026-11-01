# Verification — 2026-10-31

## Local age-estimation release 1.5 — 2026-11-01

- Downloaded 2,200 age-labeled images from a public UTKFace-derived research mirror; exact
  duplicates were removed, leaving 1,595 training, 298 validation and 299 test images.
- Trained a ridge age head on frozen RGB224 CNN features. The 299-image test MAE was 7.58 years;
  a constant training-mean baseline produced 20.27 years. Data labels are approximate, and
  identity and pretraining overlap are unknown. This is not proof of age or webcam accuracy.
- Exported ONNX, checked the graph and numerical output against the learned regressor. Real app
  image inference, custom-model diagnostics, REST inference, and a 24-frame video with 36 age
  outputs passed. Visually inspected the age label on `outputs/age-training-2026-11-01/preview.png`.
- Unit tests cover invalid age outputs and quality rejection. Age is null unless configured
  and for rejected faces. The application does not automatically save identifying names, face embeddings, images or age logs.



## Local trained-model release 1.4 — 2026-11-01

- Expanded expression training: 22,820 accepted training faces; selected epoch 66, stopped at 81.
  Fixed test raw top-1 accuracy increased from the previous candidate's 75.81% to 79.39%.
  Accuracy with uncertain predictions counted incorrect increased from 68.68% to 73.71%.
  The test subset is reused from the earlier report; this is not an external generalization test.
- The exported expression model passed ONNX graph validation and numerical comparison with
  the learned NumPy classifier. It also passed application image inference and a 24-frame
  video smoke with 36 face detections. Backbone weights remained frozen.
- Downloaded and checksum-verified the original COCO train annotation JSON; selected and
  decoded 1,588 native training images. The subset contains all 80 categories and 15,296 boxes.
  Exact pixel duplicate exclusion and strict box validation passed.
- Quality-focal gradient passed a finite-difference derivative test. Object training stopped
  after 10 epochs without a validation gain. The candidate failed the predeclared promotion
  gate and the pinned original detector remains active.
- Actual custom-object-manifest loading passed image inference together with the trained
  expression model. Modified weights or incorrect class mappings are rejected before loading.
  Diagnostics checks the chosen custom manifests without opening a camera.
- Original models are preserved; trained expression weights are installed separately and
  selected by `run-trained.sh`. Models, datasets, outputs, and local manifests are excluded
  from source packages. See TRAINING.md and the local JSON reports for the actual limits.

## Release 1.3 — combined object and expression recognition

- Downloaded both pinned optional weights and verified exact sizes and SHA-256 hashes.
  `--setup-recognition` and the enhanced launcher's `--check` pass without opening a camera.
- Actual NanoDet ONNX inference found bicycle, car and dog in the public
  [YOLOX dog/bicycle test photograph](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/assets/dog.jpg).
  The vehicle label is the model's generic class, not a verified fine-grained vehicle type.
  Real graph inspection confirmed six outputs (three score heads then three distance heads).
  The adapter follows that ordering; synthetic decoder fixtures match this verified contract.
- Actual local Qwen3-VL returned the dog, bicycle, vehicle and tree. Crop inspection also
  returned the dog and translated its box into original-frame coordinates. These examples
  demonstrate working inference, not exhaustive recognition or localization accuracy.
- HSEmotion processed the reference face with all eight finite probabilities, no alignment,
  three-view augmentation and null AU/VA outputs. Its label differed from FER+; this is not
  evidence that either label is correct. The 24-frame video fixture processed 36 face detections
  with the enhanced models in 1.81 seconds (no Qwen, includes rendering/export overhead).
- Warm CPU medians over ten repetitions on this machine: NanoDet **19.6 ms** for the 768×576
  photograph; YuNet plus HSEmotion with TTA **33.1 ms** for the one-face reference image.
  These are small smoke timings, not a general FPS benchmark or a camera latency guarantee.
- Visually checked `outputs/recognition-v1.3-objects.png` and `outputs/recognition-v1.3-face.png`:
  named object boxes and the eight-score expression panel render together correctly.
  Actual REST `/analyze` and WebSocket inference returned object boxes and HSEmotion results.
- **90 tests pass**, including decoding/letterboxing, RGB normalization, class order, source
  fusion, crop coordinate restoration, margin abstention in tracking/evaluation, configuration
  rejection, download no-storage handling and CLI/API operation without a language model.
  Ruff lint/format, mypy across **21 modules**, lock verification and wheel/sdist build pass.
  Package contents include new modules and third-party notices; weights and local media are excluded.

Physical webcam accuracy, demographic performance and calibration gains remain unmeasured.
No labeled evaluation dataset was supplied. Core FER+ defaults remain available for comparison;
the enhanced preset is a model choice, not a promise to recognize every object or expression.

## Release 1.2 — named object boxes

- The real LM Studio Qwen3-VL model returned structured names and normalized object coordinates.
  The application validated these and mapped them into original-frame pixels. Visually checked
  `outputs/object-boxes-preview.png`: cyan rectangles surround the test shapes/text, with names
  below each rectangle. The locations are approximate, not accuracy-certified detections.
- Real model inference plus a simulated moving frame stream passed: delayed source-frame boxes
  were replayed through optical flow to frame 32. A 30-pixel image translation moved the red-circle
  box by 30 pixels and the blue-square box by 32 pixels. Visually checked the resulting
  `outputs/object-motion-preview.png`; no physical webcam was used for this test.
- **80 tests passed**, including normalized coordinate mapping, malformed/out-of-range box
  rejection, truncated structured output, text-only compatibility, delayed-motion replay,
  disappearance on blank/expired/missing-history/changed-resolution frames, label placement
  below rectangles, and disabling annotations with `--no-overlays`.
- Ruff lint/format passed and mypy passed all **19 implementation modules**. The existing
  upstream Starlette/httpx test-client deprecation warning remains non-fatal.
- `run-vision.sh` now samples every two seconds when idle. Explicit CLI arguments can override
  this; box tracking expires after five seconds of source time and stores at most 180 small
  grayscale frames. No new trained model or dependency was downloaded.

## Release 1.1 — LM Studio vision

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
confidence, reliable identity, feelings, AU or VA outputs. See VISION.md.

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

## Expanded age run 1.5.1 — 2026-11-01

- Completed all 8,469 requested source rows, reusing cached images; 7,531 training examples
  remained after exclusions. The original 298/299 validation/test membership was preserved.
- Validation-only model selection chose a Huber regression head. Original-test MAE dropped
  from 7.58 to 7.01 years; the 244-image new holdout dropped from 8.23 to 7.21 years.
- The candidate passed the predeclared promotion rule and was installed via an atomic manifest
  update. The previous weights are preserved. ONNX export/numerical checks, application image
  inference and custom model diagnostics passed. No inference backbone weights changed.
- Added reference-split regression coverage. Local metrics compare approximate dataset age
  labels and do not establish person separation, demographic fairness or webcam accuracy.

## Residual expression experiment — 2026-11-01

- Trained a 64-unit residual classifier using 22,820 original training faces and their flips.
  The reference dataset hash, 2,888 validation images and 2,902 test images were preserved.
- Stopped after 17 epochs; selected epoch 5. Test top-1 changed from 79.39% to 79.70%, while
  validation top-1 changed from 78.08% to 78.29%. The validation gain failed the predeclared
  0.3-point promotion threshold; the baseline remains active.
- Numerical ONNX export comparison, real app inference, immutable baseline-weight checks,
  lint, type checks and the 96-test suite passed. The age and object models were unchanged.
