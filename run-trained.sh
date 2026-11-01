#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f models/trained-expression.json ]]; then
  echo 'No local trained expression model is installed. See TRAINING.md.' >&2
  exit 1
fi
object_options=()
if [[ -f models/trained-objects.json ]]; then
  object_options=(--object-model-spec models/trained-objects.json)
fi
exec ./run-vision.sh "${object_options[@]}" --expression-model ferplus --model-spec models/trained-expression.json \
  --no-tta --no-storage "$@"
