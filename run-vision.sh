#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./run.sh --vision --vision-interval 2 --vision-max-side 1024 \
  --object-detector nanodet --expression-model hsemotion --no-align --tta \
  --expression-margin 0.08 --face-detector-size 960 "$@"
