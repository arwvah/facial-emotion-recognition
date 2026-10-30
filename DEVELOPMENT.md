# Development, models and evaluation

## Checks and packaging

```bash
uv sync --locked --extra api --extra optimize
uv run --extra api --extra optimize ruff check .
uv run --extra api --extra optimize ruff format --check .
uv run --extra api --extra optimize mypy
uv run --extra api --extra optimize python -m unittest discover -s tests -v
uv build
```

GitHub Actions runs the same lint, format, full-module type and unit/API checks on push/PR,
without downloading production weights. Synthetic ONNX test fixtures validate contracts and
software behavior; they are not trained models or accuracy evidence. Core implementation is
in `expression_lab/`; root Python files preserve earlier launch/import paths. The wheel only
installs the namespaced package and console entry points.

## Dataset format and evaluation

Create a CSV manifest referencing image files (FER+/AffectNet-style cropped faces, or single-face
frames). Paths are relative to the CSV. Labels are canonical names; happiness/sadness/anger
aliases are accepted. Do not mix annotation/class-order conventions between datasets.

```csv
path,label,split,subject_id,group
faces/calibration-01.png,happy,calibration,subject-a,optional-supplied-group
faces/test-01.png,sad,test,subject-b,optional-supplied-group
```

The paths above are examples, not supplied files. Use genuinely separate calibration/test images,
and separate subjects when IDs are available. Duplicate paths, identical images across splits,
and shared supplied subject IDs across splits are rejected. `group` and `subject_id` are optional;
never invent demographic labels. Calibration requires explicit `split=calibration`; evaluation
uses only `test` (the default split when omitted).

```bash
expression-lab-evaluate evaluate --manifest /data/evaluation.csv --output outputs/evaluation.json
expression-lab-evaluate compare --manifest /data/evaluation.csv \
  --models benchmark.example.json --output outputs/comparison.json
expression-lab-evaluate calibrate --manifest /data/evaluation.csv \
  --output outputs/temperature.json
expression-lab-evaluate evaluate --manifest /data/evaluation.csv \
  --calibration outputs/temperature.json --output outputs/calibrated-test.json
expression-lab --video clip.mp4 --no-align --calibration outputs/temperature.json
```

Default evaluation input is a pre-cropped face, with no alignment or quality filtering. For full
frames use `--input-mode frame --align`; the largest detected face is evaluated. Crop mode cannot
perform landmark alignment. Match deployment alignment/TTA when fitting temperature. `--tta`
and `--quality-checks` are explicit benchmark settings. Frame-mode quality policies still apply.
Calibration artifacts bind the exact model hash, input contract, alignment and TTA. Fitting
minimizes calibration-split NLL, with T=1 retained if fitting cannot improve it. Assess test NLL/ECE
and accuracy independently; default T=1 and manual temperatures are not trained calibration.

Reports include an 8×9 confusion matrix (uncertain column), per-class precision/recall/F1/support,
raw top-1 and thresholded accuracy, accepted-frame coverage/selective accuracy, NLL, Brier and ECE,
plus p50/p95 latency and FPS including decode/preprocessing/inference, excluding model loading.
Macro F1/balanced accuracy include classes actually present and list all supports. These are
classifier/frame benchmarks, not guaranteed live-camera/display FPS.

Group metrics use supplied `group` annotations with `--min-group` (default 20) samples per group.
Absent annotations yield `not_assessed_no_group_annotations`; small groups yield
`insufficient_samples`. Report class balance, uncertainty and sample sizes when interpreting
any gap. No real-world accuracy or fairness results are bundled.

## Model contracts and auxiliary heads

`model.ferplus.json` shows the actual current CNN contract. A different pretrained CNN requires
its own path, SHA-256, labels/order, dimensions, color, scale, mean/std and optional output name.
Only a unique subset of the eight canonical labels is supported; untrained labels keep zero mass.
Inputs are one float32 NCHW tensor. Set `probabilities: true` only if the output already contains
normalized probabilities; otherwise logits are softmaxed. OpenVINO XML inputs additionally
require `weights_sha256` for the paired BIN. Runtime/model selection is explicit; no downloaded
weights are executed as Python or loaded with pickle.

`auxiliary.example.json` is an illustrative template requiring your **trained** ONNX model,
real checksum, matching input preprocessing, and exact named outputs. Replace every placeholder;
the sample ImageNet normalization is an example, not evidence that your model uses it. An auxiliary
model may expose VA, AUs, or both. VA labels must be `valence`/`arousal`; AU labels use FACS numbers
such as AU01/AU12. Declare output ranges and optional sigmoid. Shape/range violations fail rather
than clipping or inventing values. The model must be compatible with the app's aligned/plain crop.

```bash
expression-lab --auxiliary /models/trained-auxiliary.json --video clip.mp4
```

There are no trained auxiliary weights bundled and no emotion-to-VA/AU heuristic fallback.
Default fields are null. The selected-face view shows available VA and the strongest two AUs;
JSON contains all supplied outputs. Their clinical/psychological validity is not established.

## INT8 and OpenVINO

```bash
expression-lab-optimize openvino --output outputs/ferplus.xml
expression-lab --engine openvino --model-spec outputs/ferplus.json --video clip.mp4
expression-lab-optimize quantize --calibration /data/calibration.csv \
  --evaluation /data/test.csv --output outputs/ferplus-int8.onnx --max-accuracy-drop 0.01
```

Quantization calibration CSVs need `path,split` and `split=calibration`; labels are not required
for static range calibration. Test CSVs require labels. Quantization uses static QDQ for Conv,
MatMul and Gemm, with conversion to a suitable opset. Calibration/test paths, image hashes, and
supplied subject IDs are checked for overlap. A validation JSON compares held-out top-1 accuracy
and timing, with an explicit maximum accuracy drop. Without test data, `accepted` is null and
status is `requires_held_out_evaluation`; a failed gate returns nonzero. No command replaces the
default model automatically. Inspect per-class changes and calibration in addition to top-1.

The local conversion smoke candidate uses one public image and is **not approved** for deployment.
It verifies conversion/inference mechanics only. OpenVINO IR export preserves FP32 weights and
writes a paired-checksum model manifest. ONNX Runtime GPU options require a provider-enabled
runtime and hardware, installed separately from the default CPU wheel; don't layer conflicting
CPU/GPU wheels. Use the console command directly in that environment instead of re-syncing the
CPU lockfile. This machine's GPU path is untested.

## API configuration and hooks

The API accepts up to 8 MiB encoded images and checks decoded size at four megapixels. Model
access is serialized; WebSocket trackers/histories are per connection. REST calls are independent
still frames. Set API tokens in the environment, not in source or URLs. Browser origins other than
localhost are refused on inference endpoints. This is a local research service, not an audited
public hosting platform.

For programmatic local hooks, instantiate `EventDetector(..., hook=callable)`; emitted dictionaries
contain type, track ID, expression, duration, timestamp and confidence. CLI webhooks use the same
metadata, a 16-item queue, three-second request timeout, cooldown, and no redirect forwarding.
Timelines retain at most 1,000 recent event records and a configurable per-track history.

## Docker

```bash
docker build -t expression-lab .
docker volume create expression-models
docker run --rm -v expression-models:/data expression-lab --setup
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v expression-models:/data:ro -v "$PWD/inputs:/input:ro" -v "$PWD/outputs:/results" \
  expression-lab --video /input/clip.mp4 --headless --output /results/annotated.mp4
```

Create the host input/output folders first. The image defaults to a non-root user; the example
uses the host UID/GID for writable output mounts. The API extra is installed in the image; to
serve it, override the entry point to `expression-lab-api`. Remote container binding needs the
explicit remote flag and a token, and publish only to the intended host interface.

Docker uses hash-locked `requirements-api.txt`, generated from uv.lock. Regenerate after dependency
changes with `uv export --locked --extra api --no-dev --no-emit-project --format requirements-txt
--output-file requirements-api.txt`. Docker build/run could not be tested locally because access
to the daemon is denied and passwordless sudo is unavailable. Wheel installation and API/CLI
checks are verified independently.
