# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Download the five pinned robomimic v0.1 PH low-dim BC-RNN checkpoints and verify them.

    python download_model_zoo.py <checkpoint-dir>

Files already present with the pinned SHA-256 are kept. Downloads are sequential: the
Stanford server answers parallel requests with HTTP 503. The model zoo publishes no license
of its own; read the robomimic terms before using the checkpoints.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_zoo import CHECKPOINTS, file_sha256  # noqa: E402

ATTEMPTS = 5


def fetch(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".partial")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as handle:
                while block := response.read(1 << 20):
                    handle.write(block)
            partial.replace(target)
            return
        except OSError as exc:
            if attempt == ATTEMPTS:
                raise
            print(f"retrying {url} after {exc}", file=sys.stderr)
            time.sleep(5 * attempt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in CHECKPOINTS:
        target = args.checkpoint_dir / checkpoint.filename
        if not (target.is_file() and file_sha256(target) == checkpoint.sha256):
            print(f"downloading {checkpoint.url}")
            fetch(checkpoint.url, target)
        actual = file_sha256(target)
        if actual != checkpoint.sha256:
            target.unlink()
            raise SystemExit(f"{checkpoint.filename} has SHA-256 {actual}, expected {checkpoint.sha256}")
        print(f"verified {target} {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
