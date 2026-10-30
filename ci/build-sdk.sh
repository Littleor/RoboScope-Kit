#!/usr/bin/env bash
set -Eeuo pipefail

KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_DIST_DIR="${SDK_DIST_DIR:-${KIT_ROOT}/sdk-dist}"
if [[ -z "${SOURCE_DATE_EPOCH:-}" ]]; then
  if [[ -n "${CI_COMMIT_TIMESTAMP:-}" ]]; then
    SOURCE_DATE_EPOCH="$(python - <<'PYTHON'
import os
from datetime import datetime

print(int(datetime.fromisoformat(os.environ["CI_COMMIT_TIMESTAMP"]).timestamp()))
PYTHON
)"
  else
    SOURCE_DATE_EPOCH="$(git -C "${KIT_ROOT}" show -s --format=%ct HEAD)"
  fi
fi
export SOURCE_DATE_EPOCH

python -m build "${KIT_ROOT}/packages/roboscope-kit" --outdir "${SDK_DIST_DIR}"
python -m twine check "${SDK_DIST_DIR}"/*.whl "${SDK_DIST_DIR}"/*.tar.gz
python - "${SDK_DIST_DIR}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifacts = sorted([*root.glob("*.whl"), *root.glob("*.tar.gz")])
with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
    for path in artifacts:
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
PY
