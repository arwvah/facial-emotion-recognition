# Privacy and interpretation

Expression Lab estimates visible expression patterns. It does not establish inner emotions,
mental health, intent, trustworthiness, or suitability for employment or education decisions.
Model scores are not calibrated accuracy probabilities unless an appropriate calibration and
held-out evaluation support that interpretation.

- Obtain consent from people in view. Startup/API documentation display a notice;
  `--consent-ack` records acknowledgement in CLI frame exports without claiming consent from
  someone else or storing their identity.
- Inference uses local model files. Model setup explicitly downloads from pinned upstream URLs;
  normal processing does not download models or upload frames.
- With `--vision`, selected full frames are JPEG-encoded in memory and sent to LM Studio on
  an explicit loopback IP. Environment proxies and redirects are disabled. There is no cloud
  fallback, tool execution, conversation history, or automatic model download. `--privacy-blur`
  pixelates detected faces before the vision request too. LM Studio is a separate process:
  its own logging/settings and OS memory/swap are outside this application's no-storage promise.
- Vision results are displayed/printed, and written only with explicit export options. The
  `/describe` route requires `vision: true`; it shares the API's token and origin restrictions.
  Its answers can be wrong, omit text, or hallucinate objects; they are not identity or diagnosis.
- The API binds to loopback by default and processes encoded frames in memory. It does not
  open the server's camera, persist frames/results, or deliver webhooks. Each WebSocket gets
  separate temporary tracks and bounded recent histories.
- CLI exports and snapshots happen only when requested. `--no-storage` rejects file-export,
  report and setup flags and disables the snapshot key. Results may still be printed to stdout;
  the caller controls redirection, shell logging, OS swap, Python caches, and screen capture.
- Webhooks are disabled by default. Enabling them requires a URL and `--no-local-only`; only
  event metadata is sent, without images. Keep `--local-only` for application-level offline
  inference. Setup/package installation necessarily requires network access unless preseeded.
- Runtime inference uses ONNX Runtime/OpenCV or OpenVINO Core. OpenVINO IR serialization uses
  Core model loading, avoiding its conversion-tool telemetry path. Review third-party runtimes
  and use OS/container network restrictions when an enforced network boundary is required.
- `--privacy-blur` pixelates detected face boxes in visual outputs. Missed detections, surrounding
  appearance, and imperfect pixelation can still identify people. It is not guaranteed anonymization.
- Track IDs represent adjacent overlapping boxes, not recognized people. Crossings, occlusion,
  detector failure and camera cuts can switch or reset them. Summaries do not establish stable
  person identity.
- Demographic evaluation uses only optional, user-supplied dataset group annotations. The app
  does not infer demographics. Small groups are reported as insufficient data, and group metric
  differences alone are not a fairness certification.

No clinical, employment, disciplinary, or other consequential decision-making workflow is
implemented. Lighting, pose, camera quality, dataset bias and individual differences affect
predictions. See MODEL_CARD.md for provenance and validation limits.
