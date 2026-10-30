"""Plan SDK publication and verify the bytes visible on official PyPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_version() -> str:
    with (ROOT / "packages/roboscope-kit/pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def metadata(version: str) -> bytes:
    url = f"https://pypi.org/pypi/roboscope-kit/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def plan(version: str) -> None:
    try:
        metadata(version)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        publish = True
    else:
        publish = False
    print(f"SDK {version}: {'publish new version' if publish else 'already on PyPI'}", file=sys.stderr)
    print(f"version={version}")
    print(f"publish={str(publish).lower()}")


def verify(version: str, dist_dir: Path) -> None:
    attempts = 18
    for attempt in range(1, attempts + 1):
        try:
            remote = json.loads(metadata(version))
            break
        except urllib.error.HTTPError as error:
            if error.code != 404 or attempt == attempts:
                raise
            print(f"PyPI metadata not visible yet; retrying in 10s ({attempt}/{attempts})")
            time.sleep(10)
    artifacts = sorted([*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")])
    if len(artifacts) != 2:
        raise ValueError("expected exactly one SDK wheel and one sdist")
    hashes = {item["filename"]: item["digests"]["sha256"] for item in remote["urls"]}
    for path in artifacts:
        if hashes.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f"published hash mismatch for {path.name}")
    print(f"SDK {version}: published wheel and sdist hashes match the checked artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "verify"))
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "sdk-dist")
    args = parser.parse_args()
    version = read_version()
    if args.command == "plan":
        plan(version)
    else:
        verify(version, args.dist_dir)


if __name__ == "__main__":
    main()
