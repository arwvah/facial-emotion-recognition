# Local expression feature fusion — 2026-11-02

This experiment combines the two already installed CNNs: HSEmotion's 1,280 features
and FER+'s 1,024 features. Both backbones remain frozen. A new linear classifier is
trained on their concatenated features and exported as one ONNX model, with the same
RGB224 input contract and eight output classes as the previous local expression model.
No PyTorch, Ultralytics, CUDA, or new pretrained-model download is required.

The FER+ branch reconstructs grayscale pixel values from the normalized RGB input and
resizes to 64x64 with linear interpolation. This differs from standalone FER+'s area
resize of an original crop. Training and deployment use the same embedded preprocessing.
Inference Dropout nodes become Identity nodes when merging the older FER+ graph.
Tests compare the preprocessing numerically with OpenCV and reject modified model files.

## Selected candidate

The head uses **22,820 training images plus their horizontal flips** (45,640 views).
The original **2,888 validation and 2,902 test** images remain unchanged, including the
earlier similarity exclusions. Feature normalization is fitted on training views only.
The starting classifier initializes the HSEmotion portion; new FER+ coefficients start
at zero. Adam training uses learning rate 0.0001 and an anchor penalty of 0.01.

Training stopped at **epoch 27**, selecting **epoch 15** by minimum validation NLL.
Validation top-1 increased from **78.08% to 78.74%**. Before training, deployment required
at least 0.3 percentage points of validation top-1 gain, no fixed-test raw top-1 or macro-F1
decrease, and no fixed-test NLL increase greater than 0.02. The candidate passed those
numerical checks. It is installed as an **optional candidate**; the faster existing model
remains the default because the raw gain is small and the additional source test regresses.

| Fixed test metric | Previous classifier | Fusion candidate |
|---|---:|---:|
| Raw top-1 accuracy | 79.39% | 79.46% |
| Accuracy counting uncertain predictions incorrect | 73.71% | 74.84% |
| Coverage | 87.73% | 88.70% |
| Macro F1 including uncertainty errors | 0.76683 | 0.77251 |
| Negative log likelihood | 0.59564 | 0.59012 |
| Mean CPU classification latency | 8.81 ms | 17.92 ms |

These gains are small: raw accuracy improves by two images on this fixed test subset.
Uncertainty uses confidence 0.5 and top-two margin 0.08. Alignment and TTA are off.
The latency comparison uses two ONNX Runtime CPU threads and one face crop, excluding
image decoding and face detection. The added CNN approximately doubles classification
time; these figures are not full webcam FPS.

Model SHA-256:
`c84315695d5a8f903c32c3b782de4c6d541c147b45753c3d461460a7812d0f2f`.
The previous classifier's hash is
`d512b155abafe7453e7fc78ebb2999b0b70872e0ca00a8d5194e2e6a7b0a7d7d`.
The standard FER+, original HSEmotion, object detector, and age model are preserved.

## Additional data and rejected experiments

Three 128-unit residual heads were trained with seeds 17, 42, and 73, plus an equal-logit
ensemble. Selection used validation NLL only. The best variant gained 0.208 validation
percentage points, below the 0.3-point gate; no new test scores were computed for them.
The commands support `--defer-test` to keep test evaluation out of candidate selection.

The additional [Jeneral/fer-2013 dataset](https://huggingface.co/datasets/Jeneral/fer-2013)
was downloaded from Hugging Face, assuming that was the intended dataset host. Its
converted Parquet revision is `a487a05e71b871e68dc5c06ac4f0efc7fd6bd804`.
The **42,395,215-byte training** and **10,595,210-byte test** files have pinned SHA-256
checksums in the fetch script. Label order is checked against the pinned Parquet metadata:
angry, disgust, fear, happy, neutral, sad, surprise. There is no contempt class in this data.

Preparation accepts at most 500 training examples per class: **3,346 images** total,
including 346 disgust examples. It skips 180 known held-out perceptual matches and 68
duplicate training byte sequences. The source's **7,178 test images** remain separate.
Perceptual matching does not establish identity separation.

An additional-data run trained on **52,332 views**, including the new images and flips.
It stopped after 12 epochs with no validation-loss improvement over the starting linear
baseline. Its selected epoch was 0. The extra-data variant was rejected using validation;
test scoring remained deferred. Its training images are **not used by the selected fusion
head**. More examples did not automatically improve the target validation set.

A separate FER-2013 source regression check compares the fixed selected candidate with
the previous model. Its configuration is fixed before scoring, and its results are in
`outputs/expression-fusion-final-2026-11-02/fer2013-comparison.json`. FER+ pretraining overlap
with this dataset is unknown, so this is not an independent generalization benchmark.
On that source test split, raw top-1 was **38.37% for the previous classifier and 37.17%
for fusion**. The fusion gain on the original subset does not establish a general gain.

A further source-domain selector was trained using original training features and the
new FER training features. Both views of each of 669 FER training images were excluded
from selector fitting and reserved for validation. A blend strength was selected from
0, 0.25, 0.5, and 1 using validation only, with limits on color-validation regression.
The selected strength was 0.5. Its original test raw top-1 was **79.36%**, below the
79.39% starting baseline, so it failed the original deployment gate and was rejected.
Blend strength was not changed after observing this test result. This source test was
used diagnostically before selector development; it is a reused evaluation set.

The last attempt fine-tuned the HSEmotion CNN's **final 1x1 convolution and classifier**
with NumPy backpropagation, keeping all earlier layers frozen. It used 8,000 balanced
original training images (1,000 per class), full validation membership, learning rate
0.00001, and a five-epoch plateau limit. Gradients passed independent finite-difference
tests, and the manual forward computation matched the ONNX model. Training stopped at
epoch 5; the starting checkpoint remained best by validation NLL. It was rejected
without another test evaluation. The actual last learned parameters are preserved in
`outputs/expression-topconv-2026-11-02/last-parameters.npz`.

## Local installation and verification

The optional candidate is installed as `models/trained-expression-fusion.json` and its
checksum-pinned ONNX file. Try it with:

```bash
./run-fusion.sh --no-vision
```

The launcher enables no-storage mode and preserves the separate age/object model options.
It passed actual image inference with one detected face. The original `run-trained.sh`
still selects the faster 79.39% model. The final decision and backup manifest are in
`outputs/expression-fusion-final-2026-11-02/promotion-decision.json` and `baseline.json`.
The local test suite now has **99 passing tests**.

## Reproduce

Start from the earlier trained linear baseline and its residual-run feature cache.
After local deployment, the preserved baseline manifest is
`outputs/expression-fusion-final-2026-11-02/baseline.json`.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/train_expression_fusion.py \
  --manifest datasets/expanded-v1/expressions/manifest.csv \
  --baseline outputs/expression-fusion-final-2026-11-02/baseline.json \
  --reference-report outputs/expression-expanded-2026-11-01/training-report.json \
  --feature-run outputs/expression-residual-2026-11-01 \
  --output outputs/expression-fusion-new --epochs 80 --patience 12

PYTHONPATH=. .venv/bin/python scripts/fetch_fer2013_dataset.py \
  --output datasets/fer2013-source
# The separate data environment uses pandas, fastparquet, NumPy and OpenCV.
# In this session, all four were installed offline from the existing uv cache.
.venv-data/bin/python scripts/prepare_fer2013_dataset.py \
  --source datasets/fer2013-source \
  --heldout-manifest datasets/expanded-v1/expressions/manifest.csv \
  --reference-report outputs/expression-expanded-2026-11-01/training-report.json \
  --output datasets/fer2013-new --per-class 500
```

To reproduce the extra-data ablation, add `--fer-feature-cache` pointing to a completed
fusion run, `--extra-manifest datasets/fer2013-new/train.csv`, and `--defer-test` to the
training command, with a fresh output directory. Cached features are verified against
their model/dataset provenance and fresh CNN inference views.

The model export passes graph checking, three validation-image probability comparisons,
and frozen-backbone initializer comparisons. Results and checkpoints remain local;
training data, model weights, and feature caches are excluded from Git, packages and
the Docker context. The optional YOLO download job remains cancelled.

Validation has been reused across experiments and the fixed test subset was reported
previously. Upstream image rights, subject separation, pretraining overlap, demographic
fairness, and camera-specific accuracy remain unverified. The mirror declares Apache-2.0;
that does not independently establish rights to every source image. Visible expression
labels do not establish a person's internal feelings.
