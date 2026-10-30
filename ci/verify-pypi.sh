#!/usr/bin/env bash
set -Eeuo pipefail

KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SDK_VERSION="$(python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' "${KIT_ROOT}/packages/roboscope-kit/pyproject.toml")"
VERIFY_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${VERIFY_ROOT}"' EXIT
unset PYTHONPATH
python -m venv "${VERIFY_ROOT}/env"

for attempt in $(seq 1 18); do
  if "${VERIFY_ROOT}/env/bin/python" -m pip download \
    --index-url https://pypi.org/simple/ \
    --no-deps --only-binary=:all: \
    --dest "${VERIFY_ROOT}/download" \
    "roboscope-kit==${SDK_VERSION}"; then
    break
  fi
  if [[ "${attempt}" -eq 18 ]]; then
    echo "PyPI package resolution failed after ${attempt} attempts" >&2
    exit 1
  fi
  rm -rf -- "${VERIFY_ROOT}/download"
  echo "PyPI package index is not ready; retrying in 10s (${attempt}/18)"
  sleep 10
done

# Install the exact downloaded SDK wheel; resolve its dependencies from official PyPI.
"${VERIFY_ROOT}/env/bin/python" -m pip install \
  --index-url https://pypi.org/simple/ "${VERIFY_ROOT}/download"/*.whl
cd "${VERIFY_ROOT}"
"${VERIFY_ROOT}/env/bin/python" - <<'PY'
import os
from importlib.metadata import version

from roboscope_kit import BenchmarkName, benchmark_names, get_benchmark_spec

assert version("roboscope-kit") == os.environ["SDK_VERSION"]
assert BenchmarkName.ROBOTWIN.value in benchmark_names()
assert get_benchmark_spec(BenchmarkName.LIBERO, "libero_10").suite == "libero_10"
print(f"Published SDK {os.environ['SDK_VERSION']}: clean-environment install and public API check pass")
PY
