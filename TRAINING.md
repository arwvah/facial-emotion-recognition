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
  --root datasets/objects --epochs 10 --output outputs/object-training-new
```

Both scripts seed sampling with 42. Library versions, floating point implementations and
input datasets can affect reproducibility. Each run produces machine-readable results and
ONNX artifacts. Neither script replaces application models automatically.
