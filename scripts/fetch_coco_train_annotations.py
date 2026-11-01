"""Fetch only the compressed COCO train instances entry, with resumable range chunks."""

import argparse
import concurrent.futures
import hashlib
import shutil
import struct
import urllib.request
import zlib
from pathlib import Path

URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
ARCHIVE_SIZE = 252907541
JSON_SIZE = 469785474
SHA256 = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
CHUNK = 2 * 1024 * 1024


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_range(start, end):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not return the requested archive range")
        data = response.read()
    if len(data) != end - start + 1:
        raise ValueError("Incomplete archive range")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/.cache"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be in [1,32]")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target = args.cache_dir / "instances_train2017.json"
    if target.exists():
        if target.stat().st_size != JSON_SIZE or checksum(target) != SHA256:
            raise ValueError(
                "Existing annotation cache is invalid; preserve/remove it before retrying"
            )
        print(f"Verified cache: {target}")
        return
    if shutil.disk_usage(args.cache_dir).free < 1024**3:
        raise ValueError("At least 1 GiB free space is required")
    tail = get_range(ARCHIVE_SIZE - 16384, ARCHIVE_SIZE - 1)
    position = tail.find(b"annotations/instances_train2017.json")
    if position < 46:
        raise ValueError("Train entry not found in archive directory")
    header = struct.unpack("<4s6H3I5H2I", tail[position - 46 : position])
    if header[0] != b"PK\x01\x02" or header[4] != 8 or header[9] != JSON_SIZE:
        raise ValueError("Unexpected archive entry contract")
    compressed, offset = header[8], header[16]
    local = struct.unpack("<4s5H3I2H", get_range(offset, offset + 29))
    if local[0] != b"PK\x03\x04":
        raise ValueError("Invalid archive local header")
    start = offset + 30 + local[-2] + local[-1]
    parts = args.cache_dir / "coco-compressed-parts"
    parts.mkdir(exist_ok=True)

    def fetch(position):
        count = min(CHUNK, compressed - position)
        path = parts / str(position)
        if path.exists() and path.stat().st_size == count:
            return
        for attempt in range(3):
            try:
                data = get_range(start + position, start + position + count - 1)
                temporary = path.with_suffix(".part")
                temporary.write_bytes(data)
                temporary.replace(path)
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise

    positions = list(range(0, compressed, CHUNK))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, position) for position in positions]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"Annotation chunks {count}/{len(positions)}", flush=True)
    decoder = zlib.decompressobj(-15)
    temporary = target.with_suffix(".pending")
    with temporary.open("wb") as stream:
        for position in positions:
            stream.write(decoder.decompress((parts / str(position)).read_bytes()))
        stream.write(decoder.flush())
    if temporary.stat().st_size != JSON_SIZE or checksum(temporary) != SHA256:
        raise ValueError("Annotation checksum failed; original cache not replaced")
    temporary.replace(target)
    shutil.rmtree(parts)
    print(f"Verified annotations: {target}", flush=True)


if __name__ == "__main__":
    main()
