#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f models/trained-expression-fusion.json ]]; then
  echo 'No local fusion candidate is installed. See FUSION_TRAINING.md.' >&2
  exit 1
fi
model_options=()
if [[ -f models/trained-age.json ]]; then
  model_options+=(--age-model-spec models/trained-age.json)
fi
if [[ -f models/trained-objects.json ]]; then
  model_options+=(--object-model-spec models/trained-objects.json)
fi
exec ./run-vision.sh "${model_options[@]}" --expression-model ferplus \
  --model-spec models/trained-expression-fusion.json --no-tta --no-storage "$@"
