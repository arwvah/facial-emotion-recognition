"""Fetch a small, immutable Hugging Face FER-2013 snapshot with pinned checksums."""

import argparse
import json
from pathlib import Path

from expression_lab.emotion import download_asset

REVISION = "a487a05e71b871e68dc5c06ac4f0efc7fd6bd804"
SOURCES = {
    "train": (42395215, "02d8d517cb5607ea2b4d4e829781efe1edc88c3796fea2c3a5ddc8ec02dde876"),
    "test": (10595210, "62363a3b17f8623db20afbe7d65a701a8e1431b53e035a09b96b8ef24bbc7700"),
}
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split, (size, digest) in SOURCES.items():
        url = (
            f"https://huggingface.co/datasets/Jeneral/fer-2013/resolve/{REVISION}"
            f"/default/{split}/0000.parquet?download=true"
        )
        print(f"Fetching {split}: {size / 1e6:.1f} MB", flush=True)
        download_asset(args.output / f"{split}.parquet", url, digest, size, f"FER-2013 {split}")
        print(f"{split}: downloaded and SHA-256 verified", flush=True)
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "dataset": "Jeneral/fer-2013",
                "revision": REVISION,
                "source": "https://huggingface.co/datasets/Jeneral/fer-2013",
                "files": {s: {"bytes": n, "sha256": h} for s, (n, h) in SOURCES.items()},
                "labels": LABELS,
                "rows": {"train": 28709, "test": 7178},
                "dataset_card_declared_license": "apache-2.0",
                "limitations": "Upstream image rights and subject separation are not independently verified; do not redistribute data with source. FER+ pretraining overlap is unknown.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
