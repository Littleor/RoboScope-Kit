#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_REF="origin/main"
OUTPUT_DIR="${KIT_ROOT}/dist/official-image"
CALVIN_EVAL_SEQUENCES_URL="https://raw.githubusercontent.com/starVLA/starVLA/074ce6546f47caf401805ccf39e5776507a94790/examples/calvin/eval_files/eval_sequences.json"
CALVIN_EVAL_SEQUENCES_SHA256="c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317"

usage() {
  cat <<'EOF'
Usage: build_official_image_bundle.sh [--ref <git-ref>] [--output-dir <directory>]

Build an immutable RoboScope official base-image bundle from one RoboScope Kit commit.
The default source is origin/main. The output contains the SDK wheel, complete
benchmark runtime sources, an installer, checksums, and an image manifest.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      SOURCE_REF="${2:?--ref requires a git ref}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a directory}"
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

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  shasum -a 256 "$1" | awk '{print $1}'
}

SOURCE_COMMIT="$(git -C "${KIT_ROOT}" rev-parse --verify "${SOURCE_REF}^{commit}")"
SOURCE_SHORT="${SOURCE_COMMIT:0:12}"
SOURCE_TIMESTAMP="$(git -C "${KIT_ROOT}" show -s --format=%ct "${SOURCE_COMMIT}")"
if SOURCE_TIME_ISO="$(date -u -d "@${SOURCE_TIMESTAMP}" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null)"; then
  :
else
  SOURCE_TIME_ISO="$(TZ=UTC date -r "${SOURCE_TIMESTAMP}" '+%Y-%m-%dT%H:%M:%SZ')"
fi
BUNDLE_NAME="roboscope-official-base-image-${SOURCE_SHORT}"

SOURCE_PATHS=(
  "packages/roboscope-kit"
  "benchmarks/libero_official/starvla_runtime"
  "benchmarks/libero_official/startbenchmark.sh"
  "benchmarks/libero_official/resetenv.sh"
  "benchmarks/libero_official/roboscope_libero_runner.py"
  "benchmarks/libero_official/roboscope_libero_autotune_eval.py"
  "benchmarks/libero_official/base_assets.json"
  "benchmarks/starvla_platform_runtime/install-image-control-runtime.sh"
  "benchmarks/starvla_platform_runtime/fetch-image-control-runtime-artifacts.sh"
  "benchmarks/starvla_platform_runtime/control-runtime-dependencies.json"
  "benchmarks/starvla_platform_runtime/calvin/benchmark-image.json"
  "benchmarks/starvla_platform_runtime/libero/benchmark-image.json"
  "benchmarks/starvla_platform_runtime/libero_plus/benchmark-image.json"
  "benchmarks/starvla_platform_runtime/metaworld/benchmark-image.json"
  "benchmarks/starvla_platform_runtime/robocasa/benchmark-image.json"
  "benchmarks/starvla_platform_runtime/robotwin/benchmark-image.json"
  "benchmarks/starvla_platform_runtime/robotwin/seed-cache-v2.tar.gz"
  "benchmarks/robotwin_official/roboscope_robotwin_launcher.py"
  "benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py"
  "benchmarks/robotwin_official/verify_seed_cache_pair.py"
)

for path in "${SOURCE_PATHS[@]}"; do
  git -C "${KIT_ROOT}" cat-file -e "${SOURCE_COMMIT}:${path}"
done

STAGING_PARENT="$(mktemp -d)"
trap 'rm -rf -- "${STAGING_PARENT}"' EXIT
STAGING_DIR="${STAGING_PARENT}/${BUNDLE_NAME}"
mkdir -p "${STAGING_DIR}/evaluation-kit" "${STAGING_DIR}/wheels"

git -C "${KIT_ROOT}" archive --format=tar "${SOURCE_COMMIT}" "${SOURCE_PATHS[@]}" \
  | tar -xf - -C "${STAGING_DIR}/evaluation-kit"
CALVIN_EVAL_SEQUENCES="${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/calvin/eval_sequences.json"
curl --fail --location --silent --show-error \
  "${CALVIN_EVAL_SEQUENCES_URL}" \
  --output "${CALVIN_EVAL_SEQUENCES}"
if [[ "$(sha256_file "${CALVIN_EVAL_SEQUENCES}")" != "${CALVIN_EVAL_SEQUENCES_SHA256}" ]]; then
  echo "CALVIN evaluation sequence checksum mismatch" >&2
  exit 4
fi

SDK_PROJECT="${STAGING_DIR}/evaluation-kit/packages/roboscope-kit"
SDK_VERSION="$(awk -F '=' '/^version[[:space:]]*=/{gsub(/[[:space:]\"]/, "", $2); print $2; exit}' "${SDK_PROJECT}/pyproject.toml")"
if [[ -z "${SDK_VERSION}" ]]; then
  echo "Unable to read SDK version from ${SDK_PROJECT}/pyproject.toml" >&2
  exit 3
fi

SOURCE_DATE_EPOCH="${SOURCE_TIMESTAMP}" uv build --wheel --clear --no-progress --no-create-gitignore \
  --out-dir "${STAGING_DIR}/wheels" "${SDK_PROJECT}"
rm -rf -- \
  "${SDK_PROJECT}/build" \
  "${SDK_PROJECT}/src/roboscope_kit.egg-info"

WHEEL_COUNT="$(find "${STAGING_DIR}/wheels" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')"
if [[ "${WHEEL_COUNT}" -ne 1 ]]; then
  echo "Expected exactly one SDK wheel, found ${WHEEL_COUNT}" >&2
  exit 4
fi
WHEEL_PATH="$(find "${STAGING_DIR}/wheels" -maxdepth 1 -type f -name '*.whl')"
WHEEL_NAME="$(basename -- "${WHEEL_PATH}")"
WHEEL_SHA256="$(sha256_file "${WHEEL_PATH}")"

install -m 0755 "${SCRIPT_DIR}/install_official_image_bundle.sh" "${STAGING_DIR}/install.sh"
install -m 0755 "${SCRIPT_DIR}/verify_official_image.sh" "${STAGING_DIR}/verify.sh"
install -m 0755 "${SCRIPT_DIR}/prepare_official_image_for_bake.sh" "${STAGING_DIR}/prepare-for-bake.sh"
install -m 0755 \
  "${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/install-image-control-runtime.sh" \
  "${STAGING_DIR}/install-control-runtime.sh"

RUNTIME_CHECKSUMS="${STAGING_DIR}/runtime.sha256"
: > "${RUNTIME_CHECKSUMS}"
while IFS= read -r source_file; do
  relative="${source_file#${STAGING_DIR}/evaluation-kit/}"
  printf '%s  %s\n' "$(sha256_file "${source_file}")" "roboscope-kit/${relative}" >> "${RUNTIME_CHECKSUMS}"
done < <(
  find \
    "${STAGING_DIR}/evaluation-kit/packages/roboscope-kit" \
    "${STAGING_DIR}/evaluation-kit/benchmarks/libero_official/starvla_runtime" \
    -type f | LC_ALL=C sort
)

while IFS= read -r source_file; do
  relative="${source_file#${SDK_PROJECT}/}"
  printf '%s  %s\n' \
    "$(sha256_file "${source_file}")" \
    "../opt/roboscope/control-runtime/packages/roboscope-kit/${relative}" \
    >> "${RUNTIME_CHECKSUMS}"
done < <(find "${SDK_PROJECT}" -type f | LC_ALL=C sort)
CONTROL_LAUNCHER="${STAGING_DIR}/evaluation-kit/benchmarks/robotwin_official/roboscope_robotwin_launcher.py"
printf '%s  %s\n' \
  "$(sha256_file "${CONTROL_LAUNCHER}")" \
  "../opt/roboscope/control-runtime/benchmarks/robotwin_official/roboscope_robotwin_launcher.py" \
  >> "${RUNTIME_CHECKSUMS}"
CONTROL_SEED_SCANNER="${STAGING_DIR}/evaluation-kit/benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py"
printf '%s  %s\n' \
  "$(sha256_file "${CONTROL_SEED_SCANNER}")" \
  "../opt/roboscope/control-runtime/benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py" \
  >> "${RUNTIME_CHECKSUMS}"
CONTROL_PAIR_VERIFIER="${STAGING_DIR}/evaluation-kit/benchmarks/robotwin_official/verify_seed_cache_pair.py"
printf '%s  %s\n' \
  "$(sha256_file "${CONTROL_PAIR_VERIFIER}")" \
  "../opt/roboscope/control-runtime/benchmarks/robotwin_official/verify_seed_cache_pair.py" \
  >> "${RUNTIME_CHECKSUMS}"
printf '%s  %s\n' \
  "$(sha256_file "${STAGING_DIR}/install-control-runtime.sh")" \
  ".roboscope-official-image/verify-control-runtime.sh" \
  >> "${RUNTIME_CHECKSUMS}"

# The per-benchmark benchmark-image.json manifests are the single source of
# truth for what runs inside a benchmark image. Ship them into the Control
# Runtime (install-image-control-runtime.sh) and checksum them here so image
# verification proves the installed manifests match this RoboScope Kit commit;
# a manual in-image edit breaks runtime.sha256 and fails verification.
for source_name in calvin libero libero_plus metaworld robocasa robotwin; do
  source_file="${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/${source_name}/benchmark-image.json"
  printf '%s  %s\n' \
    "$(sha256_file "${source_file}")" \
    "../opt/roboscope/control-runtime/benchmark-manifests/${source_name}/benchmark-image.json" \
    >> "${RUNTIME_CHECKSUMS}"
done
printf '%s  %s\n' \
  "$(sha256_file "${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/robotwin/seed-cache-v2.tar.gz")" \
  "../opt/roboscope/control-runtime/benchmark-manifests/robotwin/seed-cache-v2.tar.gz" \
  >> "${RUNTIME_CHECKSUMS}"
printf '%s  %s\n' \
  "$(sha256_file "${CALVIN_EVAL_SEQUENCES}")" \
  "../opt/roboscope/control-runtime/benchmark-manifests/calvin/eval_sequences.json" \
  >> "${RUNTIME_CHECKSUMS}"

for source_name in startbenchmark.sh resetenv.sh roboscope_libero_runner.py roboscope_libero_autotune_eval.py base_assets.json; do
  source_file="${STAGING_DIR}/evaluation-kit/benchmarks/libero_official/${source_name}"
  target_name="${source_name}"
  if [[ "${source_name}" == "base_assets.json" ]]; then
    target_name="roboscope_assets/base_assets.json"
  fi
  printf '%s  %s\n' "$(sha256_file "${source_file}")" "${target_name}" >> "${RUNTIME_CHECKSUMS}"
done

ARTIFACTS_DIR="${STAGING_DIR}/control-runtime-artifacts"
"${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/fetch-image-control-runtime-artifacts.sh" \
  --output-dir "${ARTIFACTS_DIR}" \
  --uv-lock "${SDK_PROJECT}/uv.lock"

KIT_COMMIT="${SOURCE_COMMIT}" \
KIT_REF="${SOURCE_REF}" \
KIT_COMMIT_TIME="${SOURCE_TIME_ISO}" \
SDK_VERSION="${SDK_VERSION}" \
WHEEL_NAME="${WHEEL_NAME}" \
WHEEL_SHA256="${WHEEL_SHA256}" \
ARTIFACTS_MANIFEST="${ARTIFACTS_DIR}/control-runtime-artifacts.json" \
python3 - <<'PY' > "${STAGING_DIR}/manifest.json"
import json
import os

with open(os.environ["ARTIFACTS_MANIFEST"], encoding="utf-8") as handle:
    artifacts = json.load(handle)["artifacts"]
manifest = {
    "schema_version": 1,
    "image_role": "roboscope-official-base",
    "evaluation_kit_repository": "https://github.com/MINT-SJTU/RoboScope-Kit",
    "evaluation_kit_ref": os.environ["KIT_REF"],
    "evaluation_kit_commit": os.environ["KIT_COMMIT"],
    "evaluation_kit_commit_time": os.environ["KIT_COMMIT_TIME"],
    "sdk": {
        "distribution": "roboscope-kit",
        "version": os.environ["SDK_VERSION"],
        "wheel": os.environ["WHEEL_NAME"],
        "sha256": os.environ["WHEEL_SHA256"],
    },
    "control_runtime": {
        "root": "/opt/roboscope/control-runtime",
        "python": "/opt/roboscope/control-runtime/control-python/bin/python",
        "python_version": artifacts["python"]["version"],
    },
    "control_runtime_artifacts": artifacts,
    "runtime_checksum_file": "runtime.sha256",
}
print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
PY

PAYLOAD_CHECKSUMS="${STAGING_DIR}/payload.sha256"
: > "${PAYLOAD_CHECKSUMS}"
while IFS= read -r payload_file; do
  relative="${payload_file#${STAGING_DIR}/}"
  [[ "${relative}" == "payload.sha256" ]] && continue
  printf '%s  %s\n' "$(sha256_file "${payload_file}")" "${relative}" >> "${PAYLOAD_CHECKSUMS}"
done < <(find "${STAGING_DIR}" -type f | LC_ALL=C sort)

mkdir -p "${OUTPUT_DIR}"
BUNDLE_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
STAGING_DIR="${STAGING_DIR}" \
BUNDLE_PATH="${BUNDLE_PATH}" \
SOURCE_TIMESTAMP="${SOURCE_TIMESTAMP}" \
python3 - <<'PY'
import gzip
import os
import tarfile
from pathlib import Path

source = Path(os.environ["STAGING_DIR"])
destination = Path(os.environ["BUNDLE_PATH"])
mtime = int(os.environ["SOURCE_TIMESTAMP"])
files = sorted(path for path in source.rglob("*") if path.is_file() or path.is_symlink())

with destination.open("wb") as raw_file:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, compresslevel=9, mtime=0) as gzip_file:
        with tarfile.open(fileobj=gzip_file, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for path in files:
                archive_name = f"{source.name}/{path.relative_to(source).as_posix()}"
                info = archive.gettarinfo(str(path), arcname=archive_name)
                info.uid = 0
                info.gid = 0
                info.uname = "root"
                info.gname = "root"
                info.mtime = mtime
                if info.isfile():
                    with path.open("rb") as source_file:
                        archive.addfile(info, source_file)
                else:
                    archive.addfile(info)
PY
BUNDLE_SHA256="$(sha256_file "${BUNDLE_PATH}")"
printf '%s  %s\n' "${BUNDLE_SHA256}" "$(basename -- "${BUNDLE_PATH}")" > "${BUNDLE_PATH}.sha256"

echo "Built ${BUNDLE_PATH}"
echo "RoboScope Kit commit: ${SOURCE_COMMIT}"
echo "SDK: roboscope-kit ${SDK_VERSION}"
echo "Bundle SHA-256: ${BUNDLE_SHA256}"
