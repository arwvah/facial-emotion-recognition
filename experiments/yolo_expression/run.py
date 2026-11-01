# SPDX-License-Identifier: AGPL-3.0-only
"""Install the isolated stack, train, export, and evaluate, with a persistent status file."""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch) < 1:
        parser.error("epochs, patience and batch must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    for field in ("data", "manifest", "reference_report", "baseline", "weights"):
        value = getattr(args, field).resolve()
        if not value.exists():
            raise FileNotFoundError(value)
        setattr(args, field, value)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv first")
    python = root / ".venv-yolo/bin/python"
    env = dict(os.environ)
    env.update(UV_HTTP_TIMEOUT="180", UV_HTTP_RETRIES="10", UV_CONCURRENT_DOWNLOADS="4")
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "2"
    state = {"pid": os.getpid(), "production_model_changed": False}

    def status(phase, **extra):
        state.update(phase=phase, updated_utc=dt.datetime.now(dt.UTC).isoformat(), **extra)
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.output / "status.json")
        print(json.dumps(state), flush=True)

    def run(phase, command):
        status(phase)
        subprocess.run([str(x) for x in command], check=True, cwd=root, env=env)

    phase = "setup"
    try:
        phase = "installing_dependencies"
        if not python.exists():
            run(phase, [uv, "venv", root / ".venv-yolo", "--python", "3.12"])
        run(
            phase,
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--index-url",
                "https://pypi.org/simple",
                "--require-hashes",
                "-r",
                root / "experiments/yolo_expression/requirements-lock.txt",
            ],
        )
        phase = "checking_cuda"
        run(
            phase,
            [
                python,
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                    "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
                ),
            ],
        )
        experiment = args.output / "experiment"
        phase = "training"
        run(
            phase,
            [
                python,
                "-m",
                "experiments.yolo_expression.train",
                "--data",
                args.data,
                "--weights",
                args.weights,
                "--output",
                experiment,
                "--epochs",
                args.epochs,
                "--patience",
                args.patience,
                "--batch",
                args.batch,
            ],
        )
        validation_image = next(
            p
            for p in sorted((args.data / "val").rglob("*"))
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        checkpoint = experiment / "fit/weights/best.pt"
        phase = "exporting_and_verifying"
        run(
            phase,
            [
                python,
                "-m",
                "experiments.yolo_expression.export",
                "--checkpoint",
                checkpoint,
                "--image",
                validation_image,
            ],
        )
        phase = "evaluating"
        run(
            phase,
            [
                sys.executable,
                "-m",
                "experiments.yolo_expression.evaluate",
                "--manifest",
                args.manifest,
                "--reference-report",
                args.reference_report,
                "--baseline",
                args.baseline,
                "--candidate",
                checkpoint.parent / "model.json",
                "--output",
                args.output / "evaluation.json",
            ],
        )
        report = json.loads((args.output / "evaluation.json").read_text())
        status(
            "complete",
            baseline_top1=report["baseline"]["raw_top1_accuracy"],
            yolo_top1=report["candidate"]["raw_top1_accuracy"],
        )
    except Exception as exc:
        status("failed", failed_phase=phase, error=str(exc))
        raise


if __name__ == "__main__":
    main()
