# Feature status — 1.5

| Requested area | Implementation | What remains data/hardware dependent |
|---|---|---|
| Local scene understanding | LM Studio Qwen3-VL client; frame questions, text reading, async desktop panel, timed sampling, JSONL and `/describe` API | Compatible local vision model/server; recognition is fallible and not exhaustive |
| Named object boxes | NanoDet 80-class detection each frame plus up to six Qwen regions; names below boxes, coordinate validation and delayed-box tracking | Approximate localization; no object-detection accuracy benchmark |
| Close-up inspection | R selects an original-pixel crop for Qwen; boxes map back to the full frame | Cannot recover absent details or unknown objects |
| Expression model choice | FER+ or HSEmotion RGB224, top-two ambiguity margin and configurable face detector size | Compare on representative labeled camera data |
| Approximate age | Optional local age-regression head, face overlay and API/JSON value; quality rejection | Test MAE 7.58 years on a small research subset; no verified birth records or identity-disjoint guarantee |
| Local transfer learning | Frozen CNN classifier training, validation stopping, model export and checked manifests | No perfection guarantee; object candidates rejected if they fail the evaluation gate |
| Confidence / temperature calibration | Thresholded uncertain output; held-out fitting command; model/contract-bound artifacts | Fit and assess on a labeled, representative calibration/test split |
| Face alignment | YuNet eye-line rotation and consistent eye positioning; safe crop fallback | Compare accuracy against unaligned input on the intended population |
| Lighting / quality | Size, brightness and Laplacian sharpness checks; skip/flag reasons | Tune thresholds for the camera/domain |
| Test-time augmentation | Three-view probability averaging | Compare accuracy and latency before enabling routinely |
| Swappable models/runtimes | Checksummed model contracts; ONNX Runtime, OpenCV DNN, OpenVINO CPU | Additional trained CNNs and provider-enabled GPU installations |
| Model comparison | Confusion matrix, per-class precision/recall/F1, coverage, NLL/Brier/ECE, FPS/latency | Actual FER+/AffectNet-style labeled manifests |
| Timelines / summaries | Per-track bounded history, dominant labels, transitions, offline HTML/JSON charts | Track IDs may change across gaps/crossings; no person identification |
| Multi-face dashboard | M switches to per-track cards and colored histories; Tab selects/pages | Four cards per page; all faces remain in exports |
| Valence/arousal | Validated ONNX output adapter, JSON/UI output | A compatible trained VA regression model is required; otherwise null |
| Action units | Validated FACS AU output adapter, JSON/UI output | A compatible trained AU model is required; otherwise null |
| Batch | Recursive images/videos, CSV/JSON per-file results, independent timelines, errors | Input files |
| Sustained-expression events | Local hook class, event JSONL, opt-in bounded webhooks | A configured receiver for webhook delivery |
| GUI / service | Desktop UI plus local FastAPI REST/WebSocket service and interactive API docs | No Streamlit/Gradio front end is added; FastAPI is the chosen alternative |
| Config | Strict YAML with CLI overrides and relative paths | User settings |
| Video/privacy controls | Optional overlays, detected-face pixelation, CSV/JSON-only processing, no-storage mode | Pixelation cannot anonymize missed detections |
| Packaging / Docker | Installable wheel/sdist, console commands, locked API requirements, non-root Dockerfile | Docker build/run blocked on this machine by daemon permissions |
| Skipping / async | Optical flow between detections; optional inference intervals; bounded worker prefetch | Validate missed transients and speed on the target stream |
| ONNX/OpenVINO / GPU | ONNX input adapters, paired-checksum OpenVINO IR export, provider selection | GPU/provider-specific hardware is not available for validation here |
| INT8 | Static QDQ candidate conversion, independent test-set accuracy gate, validation manifest | No production INT8 approval without representative calibration/test data |
| Engineering | 95 passing tests, full-module mypy, lint/format, GitHub Actions workflow | See the repository Actions tab for current hosted status |
| Privacy / ethics | Consent notice/ack flag, no automatic inference-time downloads, local defaults, model card | Obtain consent and assess suitability for the actual use case |

There is no default trained AU/VA model, labeled evaluation dataset, or fabricated benchmark.
The optional adapters and evaluation commands are runnable integrations; model suitability and
accuracy are separate validation requirements.
