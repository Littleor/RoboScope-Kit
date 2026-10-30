#!/usr/bin/env bash
set -Eeuo pipefail

SUITE="${2:?RoboTwin suite is required}"
QUICK_TEST="${3:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/benchmark-image.json"
CONTROL_PYTHON="${ROBOSCOPE_CONTROL_PYTHON:?ROBOSCOPE_CONTROL_PYTHON is required}"
LOCAL_ROOT="${1:?RoboTwin local runtime root is required}"
SHARED_ROOT="/root/autodl-fs/benchmarks/robotwin/runtime/root/RoboTwin"
LOCAL_ROBOTWIN="${LOCAL_ROOT}/RoboTwin"
GENERATION_ROOT="${LOCAL_ROOT}/seed-generations"
SEED_CACHE_ROOT="${LOCAL_ROOT}/seed-cache"

readarray -t identity < <("${CONTROL_PYTHON}" -I - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
protocol = manifest["benchmark_protocols"]["robotwin"]
print(protocol["robotwin_runtime_base_commit"])
print(protocol["robotwin_runtime_commit"])
print(manifest["runtime"]["seed_scanner_path"])
print(manifest["runtime"]["seed_cache_archive_path"])
print(manifest["runtime"]["seed_cache_archive_sha256"])
PY
)
BASE_COMMIT="${identity[0]}"
RUNTIME_COMMIT="${identity[1]}"
SCANNER="${identity[2]}"
SEED_ARCHIVE="${identity[3]}"
SEED_ARCHIVE_SHA256="${identity[4]}"
[[ "$(sha256sum "${SEED_ARCHIVE}" | awk '{print $1}')" == "${SEED_ARCHIVE_SHA256}" ]] || {
  echo "RoboTwin frozen seed archive differs from the image manifest" >&2
  exit 1
}

STAGING="${LOCAL_ROOT}/.RoboTwin.$$.tmp"
rm -rf -- "${STAGING}"
trap 'rm -rf -- "${STAGING}"' EXIT
git clone --shared --no-checkout "${SHARED_ROOT}" "${STAGING}"
git -C "${STAGING}" checkout --detach "${BASE_COMMIT}"
"${CONTROL_PYTHON}" -I - "${STAGING}" <<'PY'
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
tracked = subprocess.run(
    ("git", "-C", str(root), "ls-files", "-z", "*.py"),
    check=True,
    capture_output=True,
).stdout.split(b"\0")
paths = [root / value.decode() for value in tracked if value]
marker = b"from __future__ import annotations\n"
modified = 0
for path in paths:
    source = path.read_bytes()
    if marker in source:
        continue
    offset = source.find(b"\n") + 1 if source.startswith(b"#!") else 0
    path.write_bytes(source[:offset] + marker + source[offset:])
    modified += 1
if modified != 683:
    raise RuntimeError(f"RoboTwin annotation patch changed {modified} files; expected 683")
print(modified)
PY
git -C "${STAGING}" add --update
GIT_AUTHOR_NAME="RoboScope Runtime Builder" \
GIT_AUTHOR_EMAIL="runtime@localhost" \
GIT_AUTHOR_DATE="2026-08-29T00:00:00+08:00" \
GIT_COMMITTER_NAME="RoboScope Runtime Builder" \
GIT_COMMITTER_EMAIL="runtime@localhost" \
GIT_COMMITTER_DATE="2026-08-29T00:00:00+08:00" \
  git -C "${STAGING}" -c commit.gpgsign=false commit \
    -m "Apply RoboScope Python 3.8 annotation runtime patch"
ACTUAL_RUNTIME_COMMIT="$(git -C "${STAGING}" rev-parse HEAD)"
[[ "${ACTUAL_RUNTIME_COMMIT}" == "${RUNTIME_COMMIT}" ]] || {
  echo "Prepared RoboTwin commit ${ACTUAL_RUNTIME_COMMIT} differs from ${RUNTIME_COMMIT}" >&2
  exit 1
}
for relative in \
  assets/background_texture \
  assets/embodiments \
  assets/objects \
  envs/curobo
do
  source_path="${SHARED_ROOT}/${relative}"
  [[ -d "${source_path}" ]] || {
    echo "RoboTwin runtime asset is missing: ${source_path}" >&2
    exit 1
  }
  ln -s "${source_path}" "${STAGING}/${relative}"
done
[[ -z "$(git -C "${STAGING}" status --short --untracked-files=no)" ]] || {
  echo "Prepared RoboTwin tracked sources differ from ${RUNTIME_COMMIT}" >&2
  exit 1
}
rm -rf -- "${LOCAL_ROBOTWIN}"
mv "${STAGING}" "${LOCAL_ROBOTWIN}"
trap - EXIT

rm -rf -- "${SEED_CACHE_ROOT}"
tar -xzf "${SEED_ARCHIVE}" --no-same-owner -C "${LOCAL_ROOT}"

readarray -t values < <("${CONTROL_PYTHON}" -I - "${MANIFEST}" "${QUICK_TEST}" "${SUITE}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
runtime = manifest["runtime"]
defaults = manifest["defaults"]
print(runtime["seed_cache_dir"])
print(runtime.get("seed_generation_root", ""))
print(runtime.get("seed_generations", {}).get(sys.argv[3], ""))
print(1 if sys.argv[2] == "1" else int(defaults["episodes"]))
tasks = manifest["benchmark_protocols"]["robotwin"]["tasks"]
if sys.argv[2] == "1":
    tasks = tasks[:1]
for task in tasks:
    print(task)
PY
)
CACHE_DIR="${values[0]}"
[[ "${CACHE_DIR}" == "${SEED_CACHE_ROOT}" ]] || {
  echo "RoboTwin seed cache differs from the instance-local frozen input" >&2
  exit 1
}
GENERATION_DIGEST="${values[2]}"
EPISODES="${values[3]}"
TASKS=("${values[@]:4}")
[[ "${values[1]}" == "${GENERATION_ROOT}" ]] || {
  echo "RoboTwin generation root differs from the prepared runtime: ${values[1]}" >&2
  exit 1
}
if [[ "${QUICK_TEST}" != "1" && -n "${GENERATION_ROOT}" ]]; then
  if [[ -z "${GENERATION_DIGEST}" ]]; then
    echo "RoboTwin suite ${SUITE} requires a seed generation digest" >&2
    exit 1
  fi
  rm -rf -- "${GENERATION_ROOT}"
  mkdir -p "${GENERATION_ROOT}"
  "${CONTROL_PYTHON}" -I \
    -c 'from roboscope_kit.runtime.benchmarks.robotwin.seed_protocol import main; raise SystemExit(main())' \
    --source-cache-dir "${SEED_CACHE_ROOT}" \
    --generation-root "${GENERATION_ROOT}" \
    --suite "${SUITE}" \
    --benchmark-manifest "${MANIFEST}" \
    --robotwin-root "${LOCAL_ROBOTWIN}" \
    --scanner-path "${SCANNER}"
  CACHE_DIR="${GENERATION_ROOT}/${GENERATION_DIGEST}"
fi
COMMAND=(
  "${CONTROL_PYTHON}"
  -I
  -c 'from roboscope_kit.runtime.benchmarks.robotwin.seed_cache import main; raise SystemExit(main())'
  --cache-dir "${CACHE_DIR}"
  --suite "${SUITE}"
  --episodes "${EPISODES}"
)
if [[ "${#TASKS[@]}" -gt 0 ]]; then
  COMMAND+=(--tasks "${TASKS[@]}")
fi
"${COMMAND[@]}"
