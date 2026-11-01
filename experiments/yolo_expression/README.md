# Optional YOLO expression experiment

This directory is an **AGPL-3.0-only** experiment using Ultralytics. The rest of this
repository retains its existing license; this experiment is not included in the
application package. Ultralytics offers AGPL and Enterprise terms; review its
[licensing](https://www.ultralytics.com/license) before integrating or distributing
this experiment or its weights. Dataset rights are separate from software rights.

The task is eight-class classification of already cropped faces. Face detection,
object detection, and age estimation remain separate components. YOLO cannot
recognize every object or establish a person's internal feelings.

The experiment fine-tunes the entire ImageNet-pretrained YOLO26n-cls backbone.
Unlike the default crop augmentation, its transform resizes the entire face to
224x224, with training-only horizontal flips and mild color jitter. RGB pixels
are scaled to [0, 1]. The validation transform matches the application's OpenCV
linear resize. See the official [classifier documentation](https://docs.ultralytics.com/tasks/classify).

## Reproduce locally

Run from the repository root. Keep the training stack separate from the app:

```bash
uv venv .venv-yolo --python 3.12
uv pip install --python .venv-yolo/bin/python --require-hashes \
  --index-url https://pypi.org/simple -r experiments/yolo_expression/requirements-lock.txt

PYTHONPATH=. .venv/bin/python experiments/yolo_expression/prepare_data.py \
  --manifest datasets/expanded-v1/expressions/manifest.csv \
  --reference-report outputs/expression-expanded-2026-11-01/training-report.json \
  --output datasets/yolo-expressions

# Download the official weights, then verify the SHA-256 below.
mkdir -p outputs/yolo-weights
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt \
  -o outputs/yolo-weights/yolo26n-cls.pt
sha256sum outputs/yolo-weights/yolo26n-cls.pt

# Install (or verify) dependencies, train, export, check export parity, and evaluate.
# This command uses a new output directory and does not replace application models.
.venv/bin/python -m experiments.yolo_expression.run \
  --data datasets/yolo-expressions --weights outputs/yolo-weights/yolo26n-cls.pt \
  --manifest datasets/expanded-v1/expressions/manifest.csv \
  --reference-report outputs/expression-expanded-2026-11-01/training-report.json \
  --baseline models/trained-expression.json \
  --output outputs/yolo-job-new --epochs 50 --patience 8 --batch 32
```

The official `v8.4.0` asset's SHA-256 is
`0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`.
The lockfile pins the Linux/Python 3.12 training environment, including PyTorch 2.7.1
and Ultralytics 8.4.197. The CUDA stack requires approximately 3.2 GB of downloads.
No cloud training or external experiment tracking is used.

The preparation script verifies the original manifest hash and reuses the baseline's
172 similarity exclusions. It hardlinks **22,820 training** and **2,888 validation**
faces. The **2,902 test** faces are not placed in the training dataset. The best
checkpoint is selected using validation only. The original model stays active
unless a measured improvement and deployment licensing review justify replacement.

The fixed test subset has already been reported by previous experiments. Subject
separation and pretraining overlap are unknown. Results are not an independent
benchmark, a fairness evaluation, or a measurement of webcam performance.

## Job progress and results

The runner writes `status.json` with the current phase, `failed` with an error, or
`complete` with measured scores. Its `complete` state requires training, ONNX
export parity checking, and the fixed test evaluation to finish successfully.
`evaluation.json` includes accuracy, per-class scores, uncertainty coverage and
CPU latency; the benchmark excludes face detection and image decoding. Full
training metrics are in `experiment/fit/results.csv`, and `provenance.json` records
library versions, GPU, starting weight hash and split counts.

The local background job started on 2026-11-01 was **stopped at the user's request**
while installing dependencies. It never reached training. Its status is `cancelled`;
the service is inactive and will not continue the downloads. Inspect that job with:

```bash
systemctl --user status expression-yolo-training.service
cat outputs/yolo-job-2026-11-01/status.json
tail -n 20 outputs/yolo-job-2026-11-01.log
# To stop the job and its child processes:
systemctl --user stop expression-yolo-training.service
```

No YOLO expression accuracy has been measured yet. Download/install completion
does not mean training has completed. This source experiment has passed lint and
syntax checks; full training/export runtime verification is pending installation.

To inspect the selected candidate after the job succeeds, use the application's
existing ONNX adapter:

```bash
./run-trained.sh --no-vision \
  --model-spec outputs/yolo-job-2026-11-01/experiment/fit/weights/model.json
```
