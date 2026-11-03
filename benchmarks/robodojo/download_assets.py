#!/usr/bin/env python3
# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Download and verify the pinned RoboDojo simulator assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ASSET_INPUTS = json.loads(Path(__file__).with_name("asset-inputs.json").read_text(encoding="utf-8"))
REPOSITORY = ASSET_INPUTS["repository"]
REVISION = ASSET_INPUTS["revision"]
INVENTORY_SHA256 = ASSET_INPUTS["inventory_sha256"]
USER_AGENT = "roboscope-kit-robodojo/1"


def inventory(endpoint: str, inventory_path: Path | None) -> list[dict]:
    if inventory_path is not None:
        files = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        url = f"{endpoint}/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
        for attempt in range(6):
            try:
                with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=90) as response:
                    data = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError, OSError):
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 30))
        if data.get("sha") != REVISION:
            raise ValueError("RoboDojo asset metadata revision differs from the pinned revision")
        files = []
        for item in data["siblings"]:
            name = item["rfilename"]
            if not name.startswith("Assets/"):
                continue
            lfs = item.get("lfs")
            files.append({
                "path": name,
                "bytes": item["size"],
                "hash": lfs["sha256"] if lfs else item["blobId"],
                "algorithm": "sha256" if lfs else "git-blob-sha1",
            })
    if not isinstance(files, list) or not files:
        raise ValueError("RoboDojo asset inventory is empty")
    files.sort(key=lambda entry: entry["path"])
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError("RoboDojo asset inventory differs from the pinned metadata")
    for entry in files:
        name = entry["path"]
        path = PurePosixPath(name)
        if not name.startswith("Assets/") or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"invalid asset path: {name!r}")
    return files


def valid_file(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256() if entry["algorithm"] == "sha256" else hashlib.sha1()
    if entry["algorithm"] == "git-blob-sha1":
        digest.update(f"blob {entry['bytes']}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest() == entry["hash"]


def fetch(entry: dict, root: Path, endpoint: str, transport: str) -> str:
    target = root / entry["path"]
    if valid_file(target, entry):
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    primary = f"{endpoint}/datasets/{REPOSITORY}/resolve/{REVISION}/{quote(entry['path'])}"
    mirror = f"https://modelscope.cn/datasets/{REPOSITORY}/resolve/master/{quote(entry['path'])}"
    for attempt in range(8):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= entry["bytes"]:
                if valid_file(partial, entry):
                    partial.replace(target)
                    return "downloaded"
                partial.unlink(missing_ok=True)
                offset = 0
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            url = mirror if transport == "modelscope" and attempt % 2 == 0 else primary
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("asset server returned an incorrect content range")
                with partial.open("ab" if append else "wb") as stream:
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
            if not valid_file(partial, entry):
                partial.unlink(missing_ok=True)
                raise ValueError(f"asset checksum or size mismatch: {entry['path']}")
            partial.replace(target)
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, IncompleteRead) as error:
            if attempt == 7:
                raise RuntimeError(
                    f"asset download failed: {entry['path']}: {type(error).__name__}"
                ) from error
            delay = min(2**attempt, 30)
            if isinstance(error, HTTPError) and error.code == 429:
                retry_after = error.headers.get("Retry-After", "60")
                delay = max(delay, min(int(retry_after) if retry_after.isdigit() else 60, 300))
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--inventory", type=Path, help="pre-fetched pinned asset metadata")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--transport", choices=("primary", "modelscope"), default="primary")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    files = inventory(args.endpoint.rstrip("/"), args.inventory)
    root = args.root.resolve()
    print(f"RoboDojo assets: {len(files)} files, {sum(row['bytes'] for row in files)} bytes", flush=True)
    locks = {(row["algorithm"], row["hash"], row["bytes"]): threading.Lock() for row in files}
    verified: dict[tuple, Path] = {}
    stop = threading.Event()

    def download(entry: dict) -> str:
        if stop.is_set():
            return "cancelled"
        target = root / entry["path"]
        key = (entry["algorithm"], entry["hash"], entry["bytes"])
        with locks[key]:
            if stop.is_set():
                return "cancelled"
            source = verified.get(key)
            if source is not None and not valid_file(target, entry):
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                shutil.copyfile(source, partial)
                if not valid_file(partial, entry):
                    raise ValueError(f"local asset copy failed validation: {entry['path']}")
                partial.replace(target)
                return "copied"
            outcome = fetch(entry, root, args.endpoint.rstrip("/"), args.transport)
            verified[key] = target
            return outcome

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, entry) for entry in files]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 100 == 0 or count == len(files):
                    print(f"RoboDojo assets verified: {count}/{len(files)}", flush=True)
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    print("RoboDojo assets downloaded and verified", flush=True)


if __name__ == "__main__":
    main()
