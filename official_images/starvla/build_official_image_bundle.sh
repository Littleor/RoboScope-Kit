#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_REF=""
OUTPUT_DIR="${KIT_ROOT}/dist/official-image"
STARVLA_COMMIT="7e4a47360093d79dc42b883885c04fbe790c5b60"
CALVIN_EVAL_SEQUENCES_URL="https://raw.githubusercontent.com/starVLA/starVLA/074ce6546f47caf401805ccf39e5776507a94790/examples/calvin/eval_files/eval_sequences.json"
CALVIN_EVAL_SEQUENCES_SHA256="c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317"
METAWORLD_DATA_CONFIG_URL="https://raw.githubusercontent.com/starVLA/starVLA/a376315d328b7c47a18497f09ef59ecbc37e37d1/examples/simBenchmarks/MetaWorld/train_files/data_registry/data_config.py"
METAWORLD_DATA_CONFIG_SHA256="f0d9788be7400ee8d8e8ff910aec2ebb1ef2c4839c5ee82ce3dedf75a211a175"

usage() {
  cat <<'EOF'
Usage: build_official_image_bundle.sh --ref <git-ref> [--output-dir <directory>]

Build an immutable Benchmark-independent StarVLA official-image bundle from one
RoboScope Kit commit. The output contains the Policy SDK wheel, image-local
Control Runtime, StarVLA Policy runtime, installer, checksums, and image manifest.
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

if [[ -z "${SOURCE_REF}" ]]; then
  echo "--ref is required" >&2
  usage >&2
  exit 2
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  shasum -a 256 "$1" | awk '{print $1}'
}

SOURCE_COMMIT="$(git -C "${KIT_ROOT}" rev-parse --verify "${SOURCE_REF}^{commit}")"
if [[ -n "$(git -C "${KIT_ROOT}" status --short --untracked-files=no)" ]] \
  && git -C "${KIT_ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD; then
  echo "RoboScope Kit worktree has tracked changes that are not included in --ref ${SOURCE_REF}" >&2
  exit 3
fi
SOURCE_SHORT="${SOURCE_COMMIT:0:12}"
SOURCE_TIMESTAMP="$(git -C "${KIT_ROOT}" show -s --format=%ct "${SOURCE_COMMIT}")"
if SOURCE_TIME_ISO="$(date -u -d "@${SOURCE_TIMESTAMP}" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null)"; then
  :
else
  SOURCE_TIME_ISO="$(TZ=UTC date -r "${SOURCE_TIMESTAMP}" '+%Y-%m-%dT%H:%M:%SZ')"
fi
BUNDLE_NAME="roboscope-official-starvla-image-${SOURCE_SHORT}"

SOURCE_PATHS=(
  "packages/roboscope-kit"
  "official_images/starvla/startbenchmark.sh"
  "official_images/starvla/resetenv.sh"
  "official_images/starvla/roboscope-policy.json"
  "official_images/starvla/starvla_official_policy.py"
  "official_images/starvla/starvla_base_assets.py"
  "official_images/starvla/starvla_full_checkpoint.py"
  "official_images/starvla/export_starvla_eval_run.py"
  "official_images/starvla/install_official_image_bundle.sh"
  "official_images/starvla/verify_official_image.sh"
  "official_images/starvla/prepare_official_image_for_bake.sh"
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
mkdir -p \
  "${STAGING_DIR}/evaluation-kit" \
  "${STAGING_DIR}/wheels" \
  "${STAGING_DIR}/control-runtime/packages" \
  "${STAGING_DIR}/control-runtime/benchmarks/robotwin_official"

git -C "${KIT_ROOT}" archive --format=tar "${SOURCE_COMMIT}" "${SOURCE_PATHS[@]}" \
  | tar -xf - -C "${STAGING_DIR}/evaluation-kit"

SDK_PROJECT="${STAGING_DIR}/evaluation-kit/packages/roboscope-kit"
SDK_PACKAGE="${SDK_PROJECT}/src/roboscope_kit"
cp -a "${SDK_PROJECT}" "${STAGING_DIR}/control-runtime/packages/"
install -m 0755 \
  "${STAGING_DIR}/evaluation-kit/benchmarks/robotwin_official/roboscope_robotwin_launcher.py" \
  "${STAGING_DIR}/control-runtime/benchmarks/robotwin_official/roboscope_robotwin_launcher.py"
install -m 0755 \
  "${STAGING_DIR}/evaluation-kit/benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py" \
  "${STAGING_DIR}/control-runtime/benchmarks/robotwin_official/roboscope_robotwin_seed_scan.py"
install -m 0755 \
  "${STAGING_DIR}/evaluation-kit/benchmarks/robotwin_official/verify_seed_cache_pair.py" \
  "${STAGING_DIR}/control-runtime/benchmarks/robotwin_official/verify_seed_cache_pair.py"
# Ship the repo-committed per-benchmark manifests into the Control Runtime
# staging so install-image-control-runtime.sh installs them as the immutable
# reference (run-benchmark.sh aborts on any deviation).
BENCHMARK_MANIFESTS=(calvin libero libero_plus metaworld robocasa robotwin)
for source_name in "${BENCHMARK_MANIFESTS[@]}"; do
  manifest_dir="${STAGING_DIR}/control-runtime/benchmarks/starvla_platform_runtime/${source_name}"
  install -d -m 0755 "${manifest_dir}"
  install -m 0644 \
    "${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/${source_name}/benchmark-image.json" \
    "${manifest_dir}/benchmark-image.json"
done
ROBOTWIN_SEED_ARCHIVE="${STAGING_DIR}/control-runtime/benchmarks/starvla_platform_runtime/robotwin/seed-cache-v2.tar.gz"
install -m 0644 \
  "${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/robotwin/seed-cache-v2.tar.gz" \
  "${ROBOTWIN_SEED_ARCHIVE}"
CALVIN_EVAL_SEQUENCES="${STAGING_DIR}/control-runtime/benchmarks/starvla_platform_runtime/calvin/eval_sequences.json"
curl --fail --location --silent --show-error \
  "${CALVIN_EVAL_SEQUENCES_URL}" \
  --output "${CALVIN_EVAL_SEQUENCES}"
if [[ "$(sha256_file "${CALVIN_EVAL_SEQUENCES}")" != "${CALVIN_EVAL_SEQUENCES_SHA256}" ]]; then
  echo "CALVIN evaluation sequence checksum mismatch" >&2
  exit 4
fi
# The pinned in-image StarVLA runtime predates the MetaWorld example, so its
# data registry (the metaworld_mt50 mixture and metaworld_robot DataConfig) is
# shipped from a pinned upstream commit and installed into the runtime's
# examples tree by install.sh.
METAWORLD_DATA_CONFIG="${STAGING_DIR}/control-runtime/benchmarks/starvla_platform_runtime/metaworld/data_config.py"
curl --fail --location --silent --show-error \
  "${METAWORLD_DATA_CONFIG_URL}" \
  --output "${METAWORLD_DATA_CONFIG}"
if [[ "$(sha256_file "${METAWORLD_DATA_CONFIG}")" != "${METAWORLD_DATA_CONFIG_SHA256}" ]]; then
  echo "MetaWorld data registry checksum mismatch" >&2
  exit 4
fi
rm -rf -- \
  "${SDK_PACKAGE}/runtime" \
  "${SDK_PACKAGE}/benchmark_specs.py" \
  "${SDK_PACKAGE}/client.py" \
  "${SDK_PACKAGE}/policy_config.py"
cat > "${SDK_PACKAGE}/__init__.py" <<'PY'
"""Benchmark-neutral RoboScope Policy process SDK."""

from roboscope_kit.contracts import ActionSpec, BenchmarkSpec, FeatureKind, FeatureSpec
from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)
from roboscope_kit.server import PolicyServer, PolicyState, run

__all__ = [
    "ActionChunk",
    "ActionSpec",
    "BenchmarkSpec",
    "EpisodeContext",
    "FeatureKind",
    "FeatureSpec",
    "Observation",
    "Policy",
    "PolicyCapabilities",
    "PolicyServer",
    "PolicyState",
    "SessionContext",
    "run",
]
PY
python3 - "${SDK_PROJECT}/pyproject.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
output = []
skipping = False
for line in lines:
    if line.strip() == "[project.optional-dependencies]":
        skipping = True
        continue
    if skipping and line.startswith("["):
        skipping = False
    if not skipping:
        output.append(line)
path.write_text("".join(output), encoding="utf-8")
PY
SDK_VERSION="$(awk -F '=' '/^version[[:space:]]*=/{gsub(/[[:space:]\"]/, "", $2); print $2; exit}' "${SDK_PROJECT}/pyproject.toml")"
if [[ -z "${SDK_VERSION}" ]]; then
  echo "Unable to read SDK version from ${SDK_PROJECT}/pyproject.toml" >&2
  exit 3
fi

SOURCE_DATE_EPOCH="${SOURCE_TIMESTAMP}" uv build --wheel --clear --no-progress --no-create-gitignore \
  --out-dir "${STAGING_DIR}/wheels" "${SDK_PROJECT}"
rm -rf -- "${SDK_PROJECT}"

WHEEL_COUNT="$(find "${STAGING_DIR}/wheels" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')"
if [[ "${WHEEL_COUNT}" -ne 1 ]]; then
  echo "Expected exactly one SDK wheel, found ${WHEEL_COUNT}" >&2
  exit 4
fi
WHEEL_PATH="$(find "${STAGING_DIR}/wheels" -maxdepth 1 -type f -name '*.whl')"
WHEEL_NAME="$(basename -- "${WHEEL_PATH}")"
WHEEL_SHA256="$(sha256_file "${WHEEL_PATH}")"

ARTIFACTS_DIR="${STAGING_DIR}/control-runtime-artifacts"
"${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/fetch-image-control-runtime-artifacts.sh" \
  --output-dir "${ARTIFACTS_DIR}" \
  --uv-lock "${STAGING_DIR}/control-runtime/packages/roboscope-kit/uv.lock"

OFFICIAL_SOURCE="${STAGING_DIR}/evaluation-kit/official_images/starvla"
install -m 0755 "${OFFICIAL_SOURCE}/install_official_image_bundle.sh" "${STAGING_DIR}/install.sh"
install -m 0755 "${OFFICIAL_SOURCE}/verify_official_image.sh" "${STAGING_DIR}/verify.sh"
install -m 0755 \
  "${OFFICIAL_SOURCE}/prepare_official_image_for_bake.sh" \
  "${STAGING_DIR}/prepare-for-bake.sh"
install -m 0755 \
  "${STAGING_DIR}/evaluation-kit/benchmarks/starvla_platform_runtime/install-image-control-runtime.sh" \
  "${STAGING_DIR}/install-control-runtime.sh"
rm -f -- \
  "${OFFICIAL_SOURCE}/install_official_image_bundle.sh" \
  "${OFFICIAL_SOURCE}/verify_official_image.sh" \
  "${OFFICIAL_SOURCE}/prepare_official_image_for_bake.sh"
rm -rf -- "${STAGING_DIR}/evaluation-kit/benchmarks"

RUNTIME_CHECKSUMS="${STAGING_DIR}/runtime.sha256"
: > "${RUNTIME_CHECKSUMS}"
for target_name in startbenchmark.sh resetenv.sh; do
  printf '%s  %s\n' \
    "$(sha256_file "${OFFICIAL_SOURCE}/${target_name}")" \
    "${target_name}" >> "${RUNTIME_CHECKSUMS}"
done

CONTROL_SDK="${STAGING_DIR}/control-runtime/packages/roboscope-kit"
while IFS= read -r source_file; do
  relative="${source_file#${CONTROL_SDK}/}"
  printf '%s  %s\n' \
    "$(sha256_file "${source_file}")" \
    "../opt/roboscope/control-runtime/packages/roboscope-kit/${relative}" \
    >> "${RUNTIME_CHECKSUMS}"
done < <(find "${CONTROL_SDK}" -type f | LC_ALL=C sort)
CONTROL_LAUNCHER="${STAGING_DIR}/control-runtime/benchmarks/robotwin_official/roboscope_robotwin_launcher.py"
printf '%s  %s\n' \
  "$(sha256_file "${CONTROL_LAUNCHER}")" \
  "../opt/roboscope/control-runtime/benchmarks/robotwin_official/roboscope_robotwin_launcher.py" \
  >> "${RUNTIME_CHECKSUMS}"
for target_name in roboscope_robotwin_seed_scan.py verify_seed_cache_pair.py; do
  printf '%s  %s\n' \
    "$(sha256_file "${STAGING_DIR}/control-runtime/benchmarks/robotwin_official/${target_name}")" \
    "../opt/roboscope/control-runtime/benchmarks/robotwin_official/${target_name}" \
    >> "${RUNTIME_CHECKSUMS}"
done
printf '%s  %s\n' \
  "$(sha256_file "${STAGING_DIR}/install-control-runtime.sh")" \
  ".roboscope-official-image/verify-control-runtime.sh" \
  >> "${RUNTIME_CHECKSUMS}"
# Per-benchmark benchmark-image.json manifests are the single source of truth;
# checksum the installed reference so a manual in-image edit fails verification.
for source_name in "${BENCHMARK_MANIFESTS[@]}"; do
  printf '%s  %s\n' \
    "$(sha256_file "${STAGING_DIR}/control-runtime/benchmarks/starvla_platform_runtime/${source_name}/benchmark-image.json")" \
    "../opt/roboscope/control-runtime/benchmark-manifests/${source_name}/benchmark-image.json" \
    >> "${RUNTIME_CHECKSUMS}"
done
printf '%s  %s\n' \
  "$(sha256_file "${ROBOTWIN_SEED_ARCHIVE}")" \
  "../opt/roboscope/control-runtime/benchmark-manifests/robotwin/seed-cache-v2.tar.gz" \
  >> "${RUNTIME_CHECKSUMS}"
printf '%s  %s\n' \
  "$(sha256_file "${STAGING_DIR}/control-runtime/benchmarks/starvla_platform_runtime/metaworld/data_config.py")" \
  "starVLA/examples/MetaWorld/train_files/data_registry/data_config.py" \
  >> "${RUNTIME_CHECKSUMS}"
printf '%s  %s\n' \
  "$(sha256_file "${OFFICIAL_SOURCE}/roboscope-policy.json")" \
  "../workspace/roboscope-policy.json" >> "${RUNTIME_CHECKSUMS}"
printf '%s  %s\n' \
  "$(sha256_file "${OFFICIAL_SOURCE}/roboscope-policy.json")" \
  "../opt/roboscope/defaults/roboscope-policy.json" >> "${RUNTIME_CHECKSUMS}"
for source_name in \
  starvla_official_policy.py \
  starvla_base_assets.py \
  starvla_full_checkpoint.py \
  export_starvla_eval_run.py; do
  printf '%s  %s\n' \
    "$(sha256_file "${OFFICIAL_SOURCE}/${source_name}")" \
    "../opt/roboscope/starvla/${source_name}" >> "${RUNTIME_CHECKSUMS}"
done

KIT_COMMIT="${SOURCE_COMMIT}" \
KIT_REF="${SOURCE_REF}" \
KIT_COMMIT_TIME="${SOURCE_TIME_ISO}" \
SDK_VERSION="${SDK_VERSION}" \
STARVLA_COMMIT="${STARVLA_COMMIT}" \
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
    "image_role": "roboscope-starvla-official",
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
    "starvla": {
        "repository": "https://github.com/starVLA/starVLA",
        "commit": os.environ["STARVLA_COMMIT"],
        "python": "/root/miniconda3/envs/starVLA/bin/python",
        "cwd": "/root/starVLA",
        "policy": "/opt/roboscope/starvla/starvla_official_policy.py",
        "policy_config": "/workspace/roboscope-policy.json",
    },
    "run_snapshot": {
        "config": "config.yaml",
        "normalization_statistics": "dataset_statistics.json",
        "checkpoint_directory": "checkpoints",
        "selected_checkpoint_count": 1,
    },
    "python_environments": {
        "default": "/root/miniconda3/bin/python",
        "control": "/opt/roboscope/control-runtime/control-python/bin/python",
        "starvla": "/root/miniconda3/envs/starVLA/bin/python",
    },
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
