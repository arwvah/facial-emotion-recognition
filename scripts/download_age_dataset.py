import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description="Download an age-labeled research subset")
parser.add_argument("--output", type=Path, default=Path("datasets/age-training-new"))
parser.add_argument("--train-count", type=int, default=1600)
parser.add_argument("--valid-count", type=int, default=600)
parser.add_argument("--reuse", type=Path)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()
if not 1 <= args.train_count <= 7623 or not 1 <= args.valid_count <= 846:
    parser.error("Requested row counts exceed the source dataset")
if not 1 <= args.workers <= 16:
    parser.error("workers must be in [1,16]")
root = args.output
if root.exists() and not args.resume:
    raise ValueError("Choose a new output directory")
root.mkdir(parents=True, exist_ok=True)
reuse = {}
if args.reuse:
    for record in json.loads((args.reuse / "raw-manifest.json").read_text()):
        reuse[(record["source_split"], record["index"])] = record
base = "https://datasets-server.huggingface.co/rows?dataset=deedax%2FUTK-Face-Revised&config=default&split={}&offset={}&length=100"


def read(url):
    for k in range(8):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "ExpressionLab/1.4"}), timeout=40
            ) as r:
                return r.read()
        except Exception:
            if k == 7:
                raise
            time.sleep(min(2**k, 30))


tasks = [("train", i) for i in range(0, args.train_count, 100)] + [
    ("valid", i) for i in range(0, args.valid_count, 100)
]


def page(t):
    split, offset = t
    x = json.loads(read(base.format(split, offset)))
    return [(split, r["row_idx"], r["row"]["age"], r["row"]["image"]["src"]) for r in x["rows"]]


with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, args.workers)) as p:
    batches = list(p.map(page, tasks))
items = [
    item
    for batch in batches
    for item in batch
    if item[1] < (args.train_count if item[0] == "train" else args.valid_count)
]
print("Age-labelled pictures available:", len(items), flush=True)


def img(t):
    split, idx, age, url = t
    path = root / split / f"{idx:06d}.jpg"
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        old = reuse.get((split, idx))
        if old:
            previous = Path(old["path"])
            if (
                old["age"] != age
                or hashlib.sha256(previous.read_bytes()).hexdigest() != old["sha256"]
            ):
                raise ValueError("Cached dataset record changed")
            os.link(previous, path)
        else:
            temporary = path.with_suffix(".part")
            temporary.write_bytes(read(url))
            temporary.replace(path)
    return {
        "path": str(path),
        "source_split": split,
        "index": idx,
        "age": age,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as p:
    rows = []
    for i, row in enumerate(p.map(img, items), 1):
        rows.append(row)
        if i % 100 == 0:
            print("Downloaded age images:", i, "/", len(items), flush=True)
(root / "raw-manifest.json").write_text(json.dumps(rows))
print("Finished download", flush=True)
