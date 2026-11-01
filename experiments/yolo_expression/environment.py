# SPDX-License-Identifier: AGPL-3.0-only
"""Keep optional training configuration and integrations local to the experiment."""

import json
import os
from pathlib import Path


def configure(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    # Write before importing Ultralytics so its import-time telemetry is disabled too.
    (directory / "settings.json").write_text(
        json.dumps(
            {
                key: False
                for key in ("sync", "wandb", "comet", "dvc", "mlflow", "raytune", "clearml")
            }
        )
    )
    os.environ["YOLO_CONFIG_DIR"] = str(directory.resolve())
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["COMET_MODE"] = "DISABLED"
