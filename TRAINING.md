# Local transfer-learning results — 2026-11-01

Both runs train final classification layers on frozen CNN features. They do not retrain
backbones. NumPy, ONNX and ONNX Runtime are sufficient; no PyTorch installation is needed.
Production weights and the standard launcher remain unchanged. Artifacts are local, ignored
under `outputs/`, and are not distributed with source.

## Facial expressions

Run: `outputs/expression-training-2026-11-01/`.

- Trained the HSEmotion final classifier for 30 epochs; selected epoch 29 by validation NLL.
- Accepted splits: 4,745 training, 2,888 validation, 2,902 test faces. Excluded 16 perceptual
  similarity matches across splits, without modifying source images or the original manifest.
- Compared original and candidate using the same RGB224 bounding-crop preprocessing,
  no alignment and no test-time augmentation. Threshold 0.5; top-two margin 0.08.

| Test metric | Original | Trained candidate |
|---|---:|---:|
| Raw top-1 accuracy | 65.61% | 75.81% |
| Accuracy with uncertain predictions counted incorrect | 53.89% | 68.68% |
| Coverage (non-uncertain fraction) | 70.99% | 82.70% |
| Negative log likelihood | 0.9101 | 0.6712 |

These are results on this local subset. Pretraining overlap is unknown, person identity
separation is unverified, and dataset usage rights must be established separately. Validation
was used for checkpoint selection; it is not an independent temperature-calibration split.
The results do not establish webcam accuracy or improvements with the default TTA preset.

The exported ONNX model passes graph checking, matches the learned NumPy classifier's
probabilities, and works through the application's existing model manifest adapter.
Only the final classifier weights and bias changed. The original model is preserved.

Run locally:

```bash
./outputs/expression-training-2026-11-01/run-trained.sh
# Without LM Studio:
./outputs/expression-training-2026-11-01/run-trained.sh --no-vision
```

Both commands use no-storage mode. The manifest is
`outputs/expression-training-2026-11-01/model.json`; full scores, exclusions and epoch history
are in that directory's `training-report.json`.

## Object detector experiment

Run: `outputs/object-training-2026-11-01/`.

- Trained the three NanoDet classification heads for 10 epochs on 207 images.
- Backbone and bounding-box regression weights were verified unchanged.
- Validation AP50 decreased from 0.3738 to 0.2913. The original detector won model selection.
- Original test AP50 was 0.3740; the last trained candidate achieved 0.2875 and was rejected.
- `selected-model.onnx` retains the original detector; `last-trained.onnx` records the rejected
  experiment. The live app continues using the original, pinned NanoDet model.

AP here is a small custom evaluation: 101-point interpolation at IoU 0.5, confidence >= 0.05,
max 30 detections per image, ignoring crowd ground truth. It is not official COCOeval AP.
All data derives from COCO validation images; pretraining overlap is unknown. This small subset
is useful for pipeline experiments but is inadequate evidence of broad object improvements.

## Reproduce

Choose new output paths; existing runs are never overwritten. From the project root:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/train_expression_head.py \
  --manifest datasets/expressions/manifest.csv --epochs 30 --output outputs/expression-training-new
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/train_object_heads.py \
  --root datasets/objects --epochs 10 --patience 5 --output outputs/object-training-new
```

Both scripts seed sampling with 42. Library versions, floating point implementations and
input datasets can affect reproducibility. Each run produces machine-readable results and
ONNX artifacts. Neither script replaces application models automatically.

## Expanded expression run — 2026-11-01

The full source training archive was already cached and passed its pinned SHA-256 check.
Preparation accepted 22,982 faces after excluding exact duplicates. Similarity filtering in the
trainer left **22,820 training, 2,888 validation and 2,902 test** samples. Held-out membership
remained fixed; generated subject IDs were discarded because they do not establish identities.

Training began from the first trained classifier, with a maximum of 150 epochs. It stopped at
**epoch 81** after 15 epochs without validation NLL improvement of at least 0.0001. The selected
checkpoint was **epoch 66**. The backbone remained frozen.

| Fixed test metric | Previous candidate | Expanded candidate |
|---|---:|---:|
| Raw top-1 accuracy | 75.81% | 79.39% |
| Accuracy counting uncertain as incorrect | 68.68% | 73.71% |
| Coverage | 82.70% | 87.73% |
| Negative log likelihood | 0.6712 | 0.5956 |

This is a validation-selected local experiment, not a guarantee of general accuracy. The same
test split was reported in the earlier run, so it is not a previously unseen external benchmark.
Pretraining overlap, identity separation and source licensing limitations still apply.

The improved candidate was installed separately as `models/trained-expression.onnx` with a
checksum manifest. Run it with `./run-trained.sh`; add `--no-vision` to operate without LM Studio.
The launcher disables TTA to match the tested preprocessing and enables no-storage mode.
The standard launchers and original pinned models are preserved. Candidate weights and the local
manifest are excluded from Git. Full results live in `outputs/expression-expanded-2026-11-01/`.

To prepare the expanded training set in a data environment with pandas and a Parquet engine:

```bash
python scripts/expand_expression_training.py --source /tmp/affectnet_train.parquet \
  --heldout-manifest datasets/expressions/manifest.csv --output datasets/expanded-new/expressions
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/train_expression_head.py \
  --manifest datasets/expanded-new/expressions/manifest.csv \
  --initial-model-spec outputs/expression-training-2026-11-01/model.json \
  --epochs 150 --patience 15 --output outputs/expression-expanded-new
```

Training stops at the validation plateau or the explicit epoch limit. It never treats
"perfection" as an achievable or measurable stopping condition.

## Expanded object data and quality-focal experiment

The revised classifier training uses box IoU as the positive quality target with a beta=2
quality-focal loss. Its analytic logit gradient is checked against a finite-difference
calculation in the test suite. The earlier weighted-BCE experiment remains recorded above.

A validation-only pilot on the original 207 training images stopped at epoch 16, selecting
an epoch-6 checkpoint whose validation AP50 gain was negligible (approximately 0.0001).
No test-set inference was used in that pilot. This does not establish an object improvement.

COCO training annotations were retrieved from the official archive using resumable compressed
range downloads. The extracted JSON is checked against SHA-256
`610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`.
The bounded expanded subset selects up to 20 training images per category by deterministic
SHA-256 priority. Validation/test membership stays fixed at the prior 110/113 images.
The expansion preserves source category IDs, annotations, image metadata, and licenses.

Preparation needs `ijson`; expression archive preparation additionally needs pandas and a
Parquet engine such as fastparquet. These are data-preparation dependencies, separate from the
core inference dependencies. The existing `/tmp/dataset_venv` was used for those dependencies
in this local run. In a fresh preparation environment, install them explicitly.

```bash
.venv/bin/python scripts/fetch_coco_train_annotations.py
# In a preparation environment with OpenCV and ijson:
python scripts/expand_coco_training.py \
  --annotations datasets/.cache/instances_train2017.json \
  --heldout-root datasets/objects --images-per-class 20 --output datasets/objects-expanded-new
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/train_object_heads.py \
  --root datasets/objects-expanded-new --quality-focal --epochs 100 --patience 5 \
  --output outputs/object-expanded-new
```

Before expanded object training results were inspected, promotion was restricted to a validation
AP50 improvement of at least 0.005 and no test AP50 decrease. Otherwise the original detector
is retained. `--validation-only` is available for experiments that must not evaluate the test set.
The test set is evaluated after checkpoint selection; it must not guide further hyperparameter
search. As with the previous experiment, AP50 here is not the full official COCOeval metric.

The CLI/API accepts a checked custom NanoDet manifest through `--object-model-spec`. It must
contain exactly `path`, `sha256`, and `labels` in the canonical 80-category order. Custom models
must preserve the NanoDet 416×416 float32 input and output contract. `run-trained.sh` uses an
optional local `models/trained-objects.json` only when an accepted object candidate is installed.

### Expanded object result

The completed dataset contains **1,588 training images and 15,296 boxes**, all 80 categories,
with at least 23 non-crowd boxes per category. No exact pixel duplicates were excluded. The
trainer's strict box checks passed before inference. Source validation/test membership was preserved.

Training stopped after **10 epochs** without a validation improvement. Original validation
AP50 was **0.3738**; the best trained checkpoint did not beat it. The selected epoch was **0**,
meaning the original detector was retained. Its test AP50 remains **0.3740** under the stated
custom metric. No `models/trained-objects.json` was installed. Rejected trained weights remain
in `outputs/object-expanded-2026-11-01/last-trained.onnx` for inspection; they must not be
presented as an improvement.

The full run, epoch scores, per-class results, and promotion decision are saved in
`outputs/object-expanded-2026-11-01/`. The larger training set and the revised loss did not
establish an object-recognition gain. Further useful work requires a different training approach
or model architecture, followed by another independent evaluation; additional epochs alone are
not evidence of improvement. The user-visible trained launcher uses the improved expression
candidate together with the original object detector.
