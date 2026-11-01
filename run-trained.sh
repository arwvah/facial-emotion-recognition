#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f models/trained-expression.json ]]; then
  echo 'No local trained expression model is installed. See TRAINING.md.' >&2
  exit 1
fi
model_options=()
if [[ -f models/trained-age.json ]]; then
  model_options+=(--age-model-spec models/trained-age.json)
fi
if [[ -f models/trained-objects.json ]]; then
  model_options+=(--object-model-spec models/trained-objects.json)
fi
exec ./run-vision.sh "${model_options[@]}" --expression-model ferplus --model-spec models/trained-expression.json \
  --no-tta --no-storage "$@"
