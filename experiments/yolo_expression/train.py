# SPDX-License-Identifier: AGPL-3.0-only
"""Fine-tune an ImageNet-pretrained YOLO classifier locally; select on validation only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from experiments.yolo_expression.environment import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch) < 1:
        parser.error("epochs, patience and batch must be positive")
    if args.output.exists():
        raise ValueError("Choose a new experiment output directory")
    preparation = json.loads((args.data / "preparation.json").read_text())
    if (args.data / "test").exists() or not preparation[
        "test_images_not_linked_into_training_dataset"
    ]:
        raise ValueError("Do not expose the test set to the trainer")
    args.output.mkdir(parents=True)
    configure(args.output / "settings")

    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    from experiments.yolo_expression.transforms import FaceTrainer

    settings.update(
        {
            k: False
            for k in ("sync", "wandb", "comet", "mlflow", "neptune", "clearml", "hub")
            if k in settings
        }
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the local CUDA GPU")
    torch.set_num_threads(2)
    started = time.time()
    model = YOLO(str(args.weights))
    provenance = {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "starting_weights": args.weights.name,
        "starting_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "dataset": preparation,
        "selection": "Best validation top-1 checkpoint; test excluded from training and selection",
        "license": "AGPL-3.0-only experiment; see README.md",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))

    def progress(trainer):
        print(
            json.dumps(
                {
                    "epoch": trainer.epoch + 1,
                    "seconds": round(time.time() - started),
                    **trainer.metrics,
                }
            ),
            flush=True,
        )

    model.add_callback("on_fit_epoch_end", progress)
    model.train(
        trainer=FaceTrainer,
        data=str(args.data.resolve()),
        project=str(args.output.resolve()),
        name="fit",
        device=0,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=224,
        workers=0,
        cache=False,
        amp=True,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=2,
        cos_lr=True,
        dropout=0.2,
        seed=42,
        deterministic=True,
        plots=False,
        save=True,
        val=True,
        verbose=False,
    )
    print(f"Training complete: {args.output / 'fit/weights/best.pt'}", flush=True)


if __name__ == "__main__":
    main()
