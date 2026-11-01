import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description="Download an age-labeled research subset")
parser.add_argument("--output", type=Path, default=Path("datasets/age-training-new"))
root = parser.parse_args().output
if root.exists():
    raise ValueError("Choose a new output directory")
root.mkdir(parents=True)
base = "https://datasets-server.huggingface.co/rows?dataset=deedax%2FUTK-Face-Revised&config=default&split={}&offset={}&length=100"


def read(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "ExpressionLab/1.4"}), timeout=40
            ) as r:
                return r.read()
        except Exception:
            if k == 2:
                raise
            time.sleep(k + 1)


tasks = [("train", i) for i in range(0, 1600, 100)] + [("valid", i) for i in range(0, 600, 100)]


def page(t):
    split, offset = t
    x = json.loads(read(base.format(split, offset)))
    return [(split, r["row_idx"], r["row"]["age"], r["row"]["image"]["src"]) for r in x["rows"]]


with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:
    batches = list(p.map(page, tasks))
items = [item for batch in batches for item in batch]
print("Age-labelled pictures available:", len(items), flush=True)


def img(t):
    split, idx, age, url = t
    path = root / split / f"{idx:06d}.jpg"
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        path.write_bytes(read(url))
    return {
        "path": str(path),
        "source_split": split,
        "index": idx,
        "age": age,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as p:
    rows = []
    for i, row in enumerate(p.map(img, items), 1):
        rows.append(row)
        if i % 100 == 0:
            print("Downloaded age images:", i, "/", len(items), flush=True)
(root / "raw-manifest.json").write_text(json.dumps(rows))
print("Finished download", flush=True)
