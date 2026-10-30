"""Hash the executable Python sources in a platform Control Runtime."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def source_paths(root: Path) -> list[Path]:
    sdk_source = root / "packages/roboscope-kit/src"
    robotwin_sources = root / "benchmarks/robotwin_official"
    paths = list(sdk_source.rglob("*.py"))
    paths.extend(robotwin_sources.glob("*.py"))
    return sorted(path for path in paths if path.is_file())


def main() -> int:
    root = Path(sys.argv[1]).resolve(strict=True)
    digest = hashlib.sha256()
    for path in source_paths(root):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    print(digest.hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
