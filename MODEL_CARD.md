# Pretrained CNN provenance

## Recognition preset (1.3)

The enhanced launcher uses two additional local ONNX models. They are explicitly downloaded
with `--setup-recognition`, checked by size and SHA-256, and excluded from Git/packages.

| Model | Contract | Pinned source |
|---|---|---|
| NanoDet Plus | BGR416, centered black letterbox, channel mean/std normalization, distributional box regression, per-class NMS; 80 COCO object classes | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_nanodet) |
| HSEmotion `enet_b0_8_best_vgaf` | RGB224, linear resize, scale 1/255, mean [.485,.456,.406], std [.229,.224,.225]; eight logits | [Author's AffectNet models](https://github.com/av-savchenko/face-emotion-recognition/tree/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/onnx) |

- NanoDet: **3,800,954 bytes**, SHA-256 `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`.
- HSEmotion: **16,039,595 bytes**, SHA-256 `fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898`.
- HSEmotion's output order is angry, contempt, disgust, fear, happy, neutral, sad, surprise;
  the adapter reorders it into the existing canonical labels. It is an AffectNet-trained
  EfficientNet-B0 model. Preprocessing follows the [author's ONNX implementation](https://github.com/av-savchenko/hsemotion-onnx).
- Both upstream model directories/repositories carry Apache-2.0. See the bundled
  [third-party notices](expression_lab/THIRD_PARTY_NOTICES.txt); dataset terms remain separate.
- The launcher uses unaligned face bounding crops, three-view augmentation, confidence threshold
  0.5, top-two margin 0.08, and face-detection limit 960. Core FER+ defaults remain unchanged.
  Close competing predictions become `uncertain` with `ambiguous_expression`; quality rejection
  takes precedence. These heuristic thresholds have not been calibrated on a user dataset.
- NanoDet runs on CPU every processed frame and returns its raw confidence and source. Qwen
  boxes overlapping a current detection with IoU >= 0.45 are suppressed in the live view;
  distinct regions are retained. This rule prefers current geometry, not proven semantic correctness.
  Up to 30 NanoDet boxes are displayed. Detector outputs do not measure recognition certainty.
- R inspection sends an original-pixel region to Qwen. It can isolate small objects but cannot
  recover absent details or guarantee an accurate name. The two model paths have different class
  coverage and failure modes; neither recognizes every object.

A higher-resolution CNN is an additional model choice, not evidence of improved accuracy.
No labeled camera-specific comparison, fairness result, or calibration gain has been established.
Both classifiers estimate visible expressions, not internal feelings. AU/VA remain null without
separately supplied models. The comparison command supports `--expression-model hsemotion` and
`--expression-margin`; use the same held-out data to compare FER+ and HSEmotion.

## Optional scene model (1.1)

The LM Studio integration was exercised with
[lmstudio-community/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Qwen3-VL-2B-Instruct-GGUF),
Q4_K_M and its vision projector, about 1.93 GB combined according to the local LM Studio inventory.
The model is provided separately by the user and is not bundled, downloaded or checksum-managed
by Expression Lab. The repository identifies its license as Apache-2.0. The configured model
identifier is explicit; no other model or remote provider is substituted.

Inputs are a single resized JPEG plus a question; responses contain a description and up to six
approximate named boxes, without calibrated confidence or guaranteed localization/counts. Default
longest side is 768 pixels and output is limited to 512 tokens. Small text can be lost during resizing. No face crop or expression score
is required for scene questions. Answers never replace expression CNN probabilities, AU or VA outputs.
Simple shape/color and text-reading smoke checks passed; no general recognition benchmark or
claim to recognize everything is made. See [VISION.md](VISION.md).

- **Model:** FER+ Emotion Recognition, `emotion-ferplus-8.onnx`, opset 8.
- **Maintainer/source:** [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus).
- **Original training implementation:** [Microsoft FERPlus](https://github.com/microsoft/FERPlus), CNTK, cross-entropy training.
- **Paper:** Barsoum et al., [Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution](https://arxiv.org/abs/1608.01041).
- **Pinned repository commit:** `4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177`.
- **Download:** [ONNX Model Zoo mirror](https://huggingface.co/onnxmodelzoo/emotion-ferplus-8),
  pinned to `4d016bfebdb4122b1f37511c5a9d40b5c87054a8`, with the original Git LFS SHA-256
  checked after download. The mirror metadata says Apache-2.0 while its copied model
  README says MIT; consult the upstream notices before redistributing weights.
- **Size:** 35,040,571 bytes.
- **SHA-256:** `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`.
- **Input:** float32, shape `[1, 1, 64, 64]`, grayscale values in `[0,255]`.
- **Output:** eight logits: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.
- **Postprocessing:** stable eight-class softmax; map happiness → happy, sadness → sad,
  anger → angry. All eight classes are enabled; low-confidence results → uncertain.
- **License:** the upstream FER+ model README identifies this model as MIT. Weights are
  downloaded from upstream, not bundled in source. Refer to the upstream repositories for
  their license notices and to the original dataset terms for any training-data use.
- **Evaluation:** no new training or target-camera accuracy benchmark has been performed.
  A successful inference smoke test is evidence that the pipeline runs, not a measure of
  expression-recognition accuracy.

## Face detector added in 0.2

- **Model:** OpenCV Zoo `face_detection_yunet_2023mar.onnx`, via `cv2.FaceDetectorYN`.
- **Source:** [OpenCV YuNet model directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- **Pinned commit:** `f12e12798e8314f7c074a6656816c048dcc95b7a`.
- **Size:** 232,589 bytes.
- **SHA-256:** `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- **License:** MIT, per the model directory's upstream LICENSE.
- **Input:** BGR pixels, longest side capped at 640 (960 in the enhanced launcher) with aspect ratio preserved; input size
  set for each frame. This 2023 model is used with OpenCV 4.x.
- **Output:** face bounding boxes, landmarks, detection scores. Version 1.0 uses eye landmarks
  for expression-crop alignment. No face identification is performed.
- **Parameters:** detection threshold 0.85, NMS overlap threshold 0.3.
- **Fallback:** `--detector haar` uses OpenCV's bundled frontal-face cascade.

## Version 1.0 preprocessing and limitations

Eye-line alignment is enabled by default with YuNet; degenerate/missing landmarks fall back
to a bounding-box crop. Optional TTA averages original, flipped and slightly cropped inputs.
Size, lighting and sharpness checks can reject a face before classification. Rejected inputs
return `uncertain` and a quality reason; skipped CNN calls are marked `classification_skipped`.
Uniform placeholder scores on skipped calls are not evidence for any expression.

Confidence thresholds and optional fitted temperature control uncertain output. Temperature
artifacts bind model hashes, preprocessing, alignment and TTA, and must be fitted on an
independent labeled calibration split. Thresholds and manual temperatures are not proof of
calibration. EMA and frame skipping can delay changes or miss brief expressions.

No representative labeled dataset was supplied for this project. Accuracy, demographic group
performance and improvements from alignment/TTA have **not** been established. The evaluation
suite reports per-class metrics, uncertainty coverage, calibration metrics and supplied group
comparisons; it never infers demographic labels. Synthetic fixtures and the public reference
image validate software behavior only. The INT8 smoke candidate is not approved for deployment.

Valence/arousal and action-unit outputs require separately supplied, compatible trained models.
The adapters validate contracts and ranges; default outputs are null. The FER+ classifier does
not provide those heads, and expression probabilities are not converted into invented AU/VA
values. Interpret these outputs according to the supplied model's own evidence and limitations.

Visible expressions depend on context, culture, pose, lighting and individual variation. These
scores do not establish internal feelings, intent, truthfulness, mental health or suitability
for employment/education. Use consenting participants and inspect the target-domain error
patterns before relying on results. Tracking IDs are temporary spatial associations, not identity.
