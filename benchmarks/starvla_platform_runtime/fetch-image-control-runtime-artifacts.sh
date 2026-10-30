#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/control-runtime-dependencies.json"
UV_LOCK=""
OUTPUT_DIR=""

usage() {
  cat <<'EOF'
Usage: fetch-image-control-runtime-artifacts.sh --uv-lock <path> --output-dir <dir>
                                                [--manifest <path>]

Download and verify the Linux x86_64 Control Runtime install artifacts for the
official image bundles. uv and the standalone CPython archive come from the
pinned control-runtime-dependencies.json manifest; the numpy wheel is taken
from the SDK uv.lock resolved for CPython 3.11 on Linux x86_64. The fetched
artifacts make the AutoDL install fully offline.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uv-lock)
      UV_LOCK="${2:?--uv-lock requires a path}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a directory}"
      shift 2
      ;;
    --manifest)
      MANIFEST="${2:?--manifest requires a path}"
      shift 2
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

if [[ -z "${UV_LOCK}" || ! -f "${UV_LOCK}" ]]; then
  echo "SDK uv.lock is missing: ${UV_LOCK}" >&2
  exit 3
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "--output-dir is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Control Runtime dependencies manifest is missing: ${MANIFEST}" >&2
  exit 3
fi
PYTHON3="$(command -v python3 || true)"
if [[ -z "${PYTHON3}" ]]; then
  echo "python3 is required to plan the Control Runtime artifacts" >&2
  exit 3
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_sha256() {
  local path="$1" expected="$2" label="$3"
  local actual
  actual="$(sha256_file "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} SHA-256 mismatch: expected ${expected}, got ${actual}" >&2
    exit 4
  fi
}

download() {
  local url="$1" destination="$2" label="$3"
  echo "Downloading ${label}: ${url}"
  # --retry-all-errors (curl 7.71+) also retries partial-file downloads
  # (exit 18) and other transient server-side failures, not just timeouts.
  curl -fsSL --retry 3 --retry-all-errors --retry-delay 2 -o "${destination}" "${url}"
}

install -d -m 0755 "${OUTPUT_DIR}"
DOWNLOAD_PARENT="$(mktemp -d)"
trap 'rm -rf -- "${DOWNLOAD_PARENT}"' EXIT

# Plan every artifact (name, version, url, sha256, file) and write the
# bundle-local artifact manifest. numpy is resolved from the SDK uv.lock so
# the pinned wheel and its SHA-256 never drift from the lock file.
"${PYTHON3}" - \
  "${MANIFEST}" \
  "${UV_LOCK}" \
  "${DOWNLOAD_PARENT}/artifact-plan.tsv" \
  "${OUTPUT_DIR}/control-runtime-artifacts.json" <<'PY'
import json
import os
import sys

try:
    import tomllib
except ImportError:
    sys.exit("uv.lock parsing requires Python 3.11+")

manifest_path, uv_lock_path, plan_path, manifest_out_path = sys.argv[1:5]
with open(manifest_path, encoding="utf-8") as handle:
    dependencies = json.load(handle)
pinned = dependencies["artifacts"]
target = dependencies["target"]
with open(uv_lock_path, "rb") as handle:
    lock = tomllib.load(handle)

# The >=3.11 lock can express the 3.11 branch as <3.12. Select the
# concrete CPython/Linux wheel instead of depending on uv's marker spelling.
numpy_wheels = [
    {
        "version": package["version"],
        "url": wheel["url"],
        "sha256": wheel["hash"].split(":", 1)[1],
        "file": wheel["url"].split("/")[-1],
    }
    for package in lock["package"]
    if package.get("name") == "numpy"
    for wheel in package.get("wheels", [])
    if "-cp311-cp311-" in wheel["url"] and "manylinux_2_27_x86_64" in wheel["url"]
]
if len(numpy_wheels) != 1:
    sys.exit(
        "Expected exactly one CPython 3.11 manylinux x86_64 numpy wheel in uv.lock; "
        f"found {len(numpy_wheels)}"
    )
numpy = numpy_wheels[0]

artifacts = {"uv": pinned["uv"], "python": pinned["python"], "numpy": numpy}
with open(plan_path, "w", encoding="utf-8") as handle:
    for key, artifact in artifacts.items():
        handle.write(
            f"{key}\t{artifact['version']}\t{artifact['url']}\t"
            f"{artifact['sha256']}\t{artifact['file']}\n"
        )
with open(os.path.join(os.path.dirname(plan_path), "target.txt"), "w", encoding="utf-8") as handle:
    handle.write(f"{target}\n")
with open(manifest_out_path, "w", encoding="utf-8") as handle:
    json.dump(
        {"schema_version": 1, "target": target, "artifacts": artifacts},
        handle,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
PY

TARGET="$(<"${DOWNLOAD_PARENT}/target.txt")"

while IFS=$'\t' read -r name version url sha256 file; do
  case "${name}" in
    uv)
      archive="${DOWNLOAD_PARENT}/uv-${version}.tar.gz"
      download "${url}" "${archive}" "uv ${version}"
      verify_sha256 "${archive}" "${sha256}" "uv ${version}"
      tar -xzf "${archive}" -C "${DOWNLOAD_PARENT}"
      uv_binary="${DOWNLOAD_PARENT}/uv-${TARGET}/uv"
      if [[ ! -x "${uv_binary}" ]]; then
        echo "uv archive layout is unexpected: missing ${uv_binary}" >&2
        exit 4
      fi
      install -m 0755 "${uv_binary}" "${OUTPUT_DIR}/${file}"
      ;;
    python|numpy)
      download "${url}" "${OUTPUT_DIR}/${file}" "${name} ${version}"
      verify_sha256 "${OUTPUT_DIR}/${file}" "${sha256}" "${name} ${version}"
      ;;
  esac
done < "${DOWNLOAD_PARENT}/artifact-plan.tsv"

echo "Fetched Control Runtime artifacts into ${OUTPUT_DIR}:"
"${PYTHON3}" - "${OUTPUT_DIR}/control-runtime-artifacts.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    artifacts = json.load(handle)["artifacts"]
for key in ("uv", "python", "numpy"):
    artifact = artifacts[key]
    print(f"  {key} {artifact['version']}: {artifact['file']} ({artifact['sha256'][:16]}...)")
PY
