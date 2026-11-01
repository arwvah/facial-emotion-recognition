"""Optional checksum-verified local age regression model."""

import json
from pathlib import Path

import numpy as np

from .backends import InferenceBackend
from .recognition_models import hsemotion_spec


class AgeEstimator:
    def __init__(self, manifest: Path, provider="CPUExecutionProvider"):
        data = json.loads(manifest.read_text())
        if (
            not isinstance(data, dict)
            or set(data) != {"path", "sha256"}
            or not isinstance(data["path"], str)
            or not isinstance(data["sha256"], str)
        ):
            raise ValueError("Age manifest requires path and sha256")
        spec = hsemotion_spec()
        spec.path = (manifest.parent / data["path"]).resolve()
        spec.sha256 = data["sha256"]
        self.backend = InferenceBackend(spec, provider=provider)

    def predict(self, crop):
        result = np.asarray(self.backend.run(self.backend.spec.tensor(crop))).reshape(-1)
        if len(result) != 1 or not np.isfinite(result).all():
            raise ValueError("Age model must return one finite estimate")
        return float(np.clip(result[0], 0, 116))
