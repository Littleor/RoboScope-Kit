#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT=""
TARGET_ROOT="/opt/roboscope/control-runtime"
ARTIFACTS_DIR=""
KIT_COMMIT=""
SDK_VERSION=""
VERIFY_ONLY=0

usage() {
  cat <<'EOF'
Usage: install-image-control-runtime.sh [--source-root <path>]
                                        --artifacts-dir <directory>
                                        [--target-root <absolute-path>]
                                        --kit-commit <git-commit>
                                        --sdk-version <version>
                                        [--verify-only]

Install or verify the image-local RoboScope Control Runtime. Installation
requires the Linux x86_64 artifacts fetched by
fetch-image-control-runtime-artifacts.sh and runs fully offline. Verification
has no artifacts or uv dependency.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="${2:?--source-root requires a path}"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="${2:?--artifacts-dir requires a directory}"
      shift 2
      ;;
    --target-root)
      TARGET_ROOT="${2:?--target-root requires an absolute path}"
      shift 2
      ;;
    --kit-commit)
      KIT_COMMIT="${2:?--kit-commit requires a value}"
      shift 2
      ;;
    --sdk-version)
      SDK_VERSION="${2:?--sdk-version requires a value}"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${TARGET_ROOT}" != /* || "${TARGET_ROOT}" == "/" || "$(basename -- "${TARGET_ROOT}")" != "control-runtime" ]]; then
  echo "Control Runtime target must be an absolute control-runtime directory: ${TARGET_ROOT}" >&2
  exit 2
fi
if [[ ! "${KIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "RoboScope Kit commit must be a 40-character Git hash: ${KIT_COMMIT}" >&2
  exit 2
fi
if [[ -z "${SDK_VERSION}" ]]; then
  echo "SDK version is required" >&2
  exit 2
fi
if [[ "${VERIFY_ONLY}" != "1" && -z "${ARTIFACTS_DIR}" ]]; then
  echo "--artifacts-dir is required for installation" >&2
  usage >&2
  exit 2
fi

CONTROL_PYTHON="${TARGET_ROOT}/control-python/bin/python"
CONTROL_SOURCE_ROOT="${TARGET_ROOT}/packages/roboscope-kit/src"
CONTROL_LAUNCHER="${TARGET_ROOT}/benchmarks/robotwin_official/roboscope_robotwin_launcher.py"
CONTROL_SEED_SCANNER="${TARGET_ROOT}/benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py"
CONTROL_PAIR_VERIFIER="${TARGET_ROOT}/benchmarks/robotwin_official/verify_seed_cache_pair.py"
ROBOTWIN_SEED_ARCHIVE="${TARGET_ROOT}/benchmark-manifests/robotwin/seed-cache-v2.tar.gz"
CALVIN_EVAL_SEQUENCES="${TARGET_ROOT}/benchmark-manifests/calvin/eval_sequences.json"
CALVIN_EVAL_SEQUENCES_SHA256="c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317"

verify_control_runtime() {
  for path in \
    "${TARGET_ROOT}/KIT_COMMIT" \
    "${TARGET_ROOT}/SDK_VERSION" \
    "${TARGET_ROOT}/PYTHON_VERSION" \
    "${CONTROL_PYTHON}" \
    "${CONTROL_SOURCE_ROOT}/roboscope_kit/runtime/benchmark/__main__.py" \
    "${CONTROL_LAUNCHER}" \
    "${CONTROL_SEED_SCANNER}" \
    "${CONTROL_PAIR_VERIFIER}" \
    "${ROBOTWIN_SEED_ARCHIVE}" \
    "${TARGET_ROOT}/benchmark-manifests/calvin/benchmark-image.json" \
    "${CALVIN_EVAL_SEQUENCES}" \
    "${TARGET_ROOT}/benchmark-manifests/libero/benchmark-image.json" \
    "${TARGET_ROOT}/benchmark-manifests/libero_plus/benchmark-image.json" \
    "${TARGET_ROOT}/benchmark-manifests/metaworld/benchmark-image.json" \
    "${TARGET_ROOT}/benchmark-manifests/robocasa/benchmark-image.json" \
    "${TARGET_ROOT}/benchmark-manifests/robotwin/benchmark-image.json"; do
    [[ -e "${path}" ]] || return 1
  done
  [[ "$(<"${TARGET_ROOT}/KIT_COMMIT")" == "${KIT_COMMIT}" ]] || return 1
  [[ "$(<"${TARGET_ROOT}/SDK_VERSION")" == "${SDK_VERSION}" ]] || return 1
  local control_python_version
  control_python_version="$(<"${TARGET_ROOT}/PYTHON_VERSION")"
  [[ "${control_python_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  [[ -x "${CONTROL_PYTHON}" ]] || return 1
  [[ "$(sha256sum "${CALVIN_EVAL_SEQUENCES}" | awk '{print $1}')" == "${CALVIN_EVAL_SEQUENCES_SHA256}" ]] || return 1
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${CONTROL_PYTHON}" -I - \
    "${CONTROL_SOURCE_ROOT}" "${TARGET_ROOT}" "${control_python_version}" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
import sysconfig

import roboscope_kit.runtime.benchmark
from roboscope_kit.runtime.benchmarks.calvin.adapter import SDK_IMPORT_ROOT as calvin_root
from roboscope_kit.runtime.benchmarks.libero.adapter import SDK_IMPORT_ROOT as libero_root
from roboscope_kit.runtime.benchmarks.libero_plus.adapter import SDK_IMPORT_ROOT as libero_plus_root
from roboscope_kit.runtime.benchmarks.metaworld.adapter import SDK_IMPORT_ROOT as metaworld_root
from roboscope_kit.runtime.benchmarks.robocasa.adapter import SDK_IMPORT_ROOT as robocasa_root
import numpy

source_root = Path(sys.argv[1]).resolve()
target_root = Path(sys.argv[2]).resolve()
expected_python = tuple(int(part) for part in sys.argv[3].split("."))
site_packages = Path(sysconfig.get_path("purelib")).resolve()
module_path = Path(roboscope_kit.runtime.benchmark.__file__).resolve()
executable = Path(sys.executable).resolve()
assert sys.version_info[:3] == expected_python
assert executable.is_relative_to(target_root)
assert module_path.is_relative_to(source_root)
assert not source_root.is_relative_to(site_packages)
assert {calvin_root, libero_root, libero_plus_root, metaworld_root, robocasa_root} == {source_root}
robotwin_reference = target_root / "benchmark-manifests/robotwin"
robotwin_manifest = json.loads((robotwin_reference / "benchmark-image.json").read_text())
seed_digest = hashlib.sha256((robotwin_reference / "seed-cache-v2.tar.gz").read_bytes()).hexdigest()
assert seed_digest == robotwin_manifest["runtime"]["seed_cache_archive_sha256"]
PY
}

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  if ! verify_control_runtime; then
    echo "Image Control Runtime verification failed: ${TARGET_ROOT}" >&2
    exit 4
  fi
  echo "Image Control Runtime verified: ${TARGET_ROOT}"
  exit 0
fi

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "Image Control Runtime installation requires Linux x86_64" >&2
  exit 3
fi
if [[ -z "${SOURCE_ROOT}" || ! -d "${SOURCE_ROOT}" ]]; then
  echo "Control Runtime source root is missing: ${SOURCE_ROOT}" >&2
  exit 3
fi
SOURCE_SDK="${SOURCE_ROOT}/packages/roboscope-kit"
SOURCE_LAUNCHER="${SOURCE_ROOT}/benchmarks/robotwin_official/roboscope_robotwin_launcher.py"
# Per-benchmark benchmark-image.json manifests, installed into the Control
# Runtime as the immutable repo-committed reference (single source of truth)
# that run-benchmark.sh compares the live manifest against. <src>:<reference>.
BENCHMARK_MANIFESTS=(
  "calvin:calvin"
  "libero:libero"
  "libero_plus:libero_plus"
  "metaworld:metaworld"
  "robocasa:robocasa"
  "robotwin:robotwin"
)
for path in \
  "${SOURCE_SDK}/pyproject.toml" \
  "${SOURCE_SDK}/uv.lock" \
  "${SOURCE_SDK}/src/roboscope_kit/runtime/benchmark/__main__.py" \
  "${SOURCE_LAUNCHER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Control Runtime source is incomplete: ${path}" >&2
    exit 3
  fi
done
for manifest_spec in "${BENCHMARK_MANIFESTS[@]}"; do
  source_name="${manifest_spec%%:*}"
  if [[ ! -e "${SOURCE_ROOT}/benchmarks/starvla_platform_runtime/${source_name}/benchmark-image.json" ]]; then
    echo "Control Runtime benchmark manifest is missing: ${source_name}" >&2
    exit 3
  fi
done
SOURCE_CALVIN_EVAL_SEQUENCES="${SOURCE_ROOT}/benchmarks/starvla_platform_runtime/calvin/eval_sequences.json"
if [[ ! -f "${SOURCE_CALVIN_EVAL_SEQUENCES}" ]]; then
  echo "Control Runtime CALVIN evaluation sequences are missing" >&2
  exit 3
fi
if [[ "$(sha256sum "${SOURCE_CALVIN_EVAL_SEQUENCES}" | awk '{print $1}')" != "${CALVIN_EVAL_SEQUENCES_SHA256}" ]]; then
  echo "Control Runtime CALVIN evaluation sequence checksum mismatch" >&2
  exit 3
fi
SOURCE_SDK_VERSION="$(
  awk -F '=' \
    '/^version[[:space:]]*=/{gsub(/[[:space:]\"]/, "", $2); print $2; exit}' \
    "${SOURCE_SDK}/pyproject.toml"
)"
if [[ "${SOURCE_SDK_VERSION}" != "${SDK_VERSION}" ]]; then
  echo "Control Runtime SDK version mismatch: ${SOURCE_SDK_VERSION} != ${SDK_VERSION}" >&2
  exit 3
fi
ARTIFACTS_MANIFEST="${ARTIFACTS_DIR}/control-runtime-artifacts.json"
if [[ ! -f "${ARTIFACTS_MANIFEST}" ]]; then
  echo "Control Runtime artifacts manifest is missing: ${ARTIFACTS_MANIFEST}" >&2
  exit 3
fi
HOST_PYTHON="$(command -v python3 || true)"
if [[ -z "${HOST_PYTHON}" ]]; then
  echo "python3 is required to read the Control Runtime artifacts manifest" >&2
  exit 3
fi
ARTIFACT_INFO="$("${HOST_PYTHON}" - "${ARTIFACTS_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    artifacts = json.load(handle)["artifacts"]
print(artifacts["uv"]["version"])
print(artifacts["python"]["file"])
print(artifacts["numpy"]["file"])
print(artifacts["python"]["version"])
PY
)"
mapfile -t ARTIFACT_LINES <<< "${ARTIFACT_INFO}"
if [[ "${#ARTIFACT_LINES[@]}" -ne 4 ]]; then
  echo "Control Runtime artifacts manifest is incomplete: ${ARTIFACTS_MANIFEST}" >&2
  exit 3
fi
UV_VERSION="${ARTIFACT_LINES[0]}"
UV_BIN="${ARTIFACTS_DIR}/uv"
PYTHON_ARCHIVE="${ARTIFACTS_DIR}/${ARTIFACT_LINES[1]}"
NUMPY_WHEEL="${ARTIFACTS_DIR}/${ARTIFACT_LINES[2]}"
PYTHON_VERSION="${ARTIFACT_LINES[3]}"
for path in "${UV_BIN}" "${PYTHON_ARCHIVE}" "${NUMPY_WHEEL}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Control Runtime install artifact is missing: ${path}" >&2
    exit 3
  fi
done
if [[ ! -x "${UV_BIN}" ]]; then
  echo "Control Runtime uv is not executable: ${UV_BIN}" >&2
  exit 3
fi
if [[ "$("${UV_BIN}" --version)" != "uv ${UV_VERSION} "* ]]; then
  echo "Control Runtime uv version mismatch: expected uv ${UV_VERSION}" >&2
  exit 3
fi

if [[ -e "${TARGET_ROOT}" ]]; then
  if verify_control_runtime; then
    echo "Image Control Runtime is already installed: ${TARGET_ROOT}"
    exit 0
  fi
  rm -rf -- "${TARGET_ROOT}"
fi

install -d -m 0755 \
  "${TARGET_ROOT}/packages" \
  "${TARGET_ROOT}/benchmarks/robotwin_official" \
  "${TARGET_ROOT}/python" \
  "${TARGET_ROOT}/benchmark-manifests"
trap 'rm -rf -- "${TARGET_ROOT}"' ERR
cp -a "${SOURCE_SDK}" "${TARGET_ROOT}/packages/"
install -m 0755 "${SOURCE_LAUNCHER}" "${CONTROL_LAUNCHER}"
install -m 0755 \
  "${SOURCE_ROOT}/benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py" \
  "${CONTROL_SEED_SCANNER}"
install -m 0755 \
  "${SOURCE_ROOT}/benchmarks/robotwin_official/verify_seed_cache_pair.py" \
  "${CONTROL_PAIR_VERIFIER}"
for manifest_spec in "${BENCHMARK_MANIFESTS[@]}"; do
  source_name="${manifest_spec%%:*}"
  reference_name="${manifest_spec##*:}"
  reference_dir="${TARGET_ROOT}/benchmark-manifests/${reference_name}"
  install -d -m 0755 "${reference_dir}"
  install -m 0644 \
    "${SOURCE_ROOT}/benchmarks/starvla_platform_runtime/${source_name}/benchmark-image.json" \
    "${reference_dir}/benchmark-image.json"
done
install -m 0644 "${SOURCE_CALVIN_EVAL_SEQUENCES}" "${CALVIN_EVAL_SEQUENCES}"
install -m 0644 \
  "${SOURCE_ROOT}/benchmarks/starvla_platform_runtime/robotwin/seed-cache-v2.tar.gz" \
  "${ROBOTWIN_SEED_ARCHIVE}"

tar -xzf "${PYTHON_ARCHIVE}" --strip-components=1 -C "${TARGET_ROOT}/python"
if [[ "$("${TARGET_ROOT}/python/bin/python3" --version 2>&1)" != "Python ${PYTHON_VERSION}" ]]; then
  echo "Control Runtime CPython version mismatch: expected ${PYTHON_VERSION}" >&2
  exit 3
fi
"${UV_BIN}" venv \
  --python "${TARGET_ROOT}/python/bin/python3" \
  --no-python-downloads \
  "${TARGET_ROOT}/control-python"
"${UV_BIN}" pip install \
  --offline \
  --no-deps \
  --python "${CONTROL_PYTHON}" \
  "${NUMPY_WHEEL}"

CONTROL_SITE_PACKAGES="$(
  "${CONTROL_PYTHON}" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
)"
printf '%s\n' "${CONTROL_SOURCE_ROOT}" > \
  "${CONTROL_SITE_PACKAGES}/roboscope-control-runtime.pth"
printf '%s\n' "${KIT_COMMIT}" > "${TARGET_ROOT}/KIT_COMMIT"
printf '%s\n' "${SDK_VERSION}" > "${TARGET_ROOT}/SDK_VERSION"
printf '%s\n' "${PYTHON_VERSION}" > "${TARGET_ROOT}/PYTHON_VERSION"
verify_control_runtime
trap - ERR

echo "Image Control Runtime installed: ${TARGET_ROOT}"
