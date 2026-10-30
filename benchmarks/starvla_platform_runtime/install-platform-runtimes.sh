#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECOVERY_SOURCE_ROOT=""
if [[ "${1:-}" == "--recover-libero" ]]; then
  RECOVERY_SOURCE_ROOT="${2:?Usage: install-platform-runtimes.sh --recover-libero <source-root> [mount-root] [kit-commit]}"
  shift 2
fi
MOUNT_ROOT="${1:-/root/autodl-fs/benchmarks}"
SOURCE_KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
KIT_COMMIT="${2:-}"
if [[ -z "${KIT_COMMIT}" ]]; then
  KIT_COMMIT="$(git -C "${SOURCE_KIT_ROOT}" rev-parse HEAD)"
fi
if [[ ! "${KIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "RoboScope Kit commit must be a 40-character Git hash: ${KIT_COMMIT}" >&2
  exit 2
fi
SOURCE_COMMIT="$(git -C "${SOURCE_KIT_ROOT}" rev-parse HEAD)"
if [[ "${SOURCE_COMMIT}" != "${KIT_COMMIT}" ]]; then
  echo "RoboScope Kit checkout ${SOURCE_COMMIT} does not match requested commit ${KIT_COMMIT}" >&2
  exit 3
fi
if [[ -n "$(git -C "${SOURCE_KIT_ROOT}" status --short --untracked-files=no)" ]]; then
  echo "RoboScope Kit checkout has tracked changes; commit them before installing platform runtimes" >&2
  exit 3
fi
if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "Platform runtimes require Linux x86_64" >&2
  exit 3
fi

# Optional space- or comma-separated subset of destination runtime names. The
# default deploys every runtime; a subset leaves the other benchmark directories
# untouched, which the per-benchmark shared-disk recovery flows require.
REQUESTED_RUNTIMES=()
if [[ -n "${ROBOSCOPE_INSTALL_RUNTIMES:-}" ]]; then
  read -r -a REQUESTED_RUNTIMES <<< "${ROBOSCOPE_INSTALL_RUNTIMES//,/ }"
fi

runtime_selected() {
  local destination_name="$1"
  local candidate
  if [[ "${#REQUESTED_RUNTIMES[@]}" -eq 0 ]]; then
    return 0
  fi
  for candidate in "${REQUESTED_RUNTIMES[@]}"; do
    if [[ "${candidate}" == "${destination_name}" ]]; then
      return 0
    fi
  done
  return 1
}

recover_libero() {
  local source_root="$1"
  local destination="${MOUNT_ROOT}/libero"
  local staging
  local source_python="${source_root}/miniconda3/envs/libero/bin/python"
  if [[ "${source_root}" != /* || "${MOUNT_ROOT}" != /* ]]; then
    echo "Recovery source and mount root must be absolute paths" >&2
    exit 2
  fi
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    echo "Recovery requires a missing LIBERO runtime: ${destination}" >&2
    exit 4
  fi
  [[ -x "${source_python}" ]]
  [[ -f "${source_root}/LIBERO/libero/libero/__init__.py" ]]
  for directory in bddl_files init_files assets; do
    [[ -d "${source_root}/LIBERO/libero/libero/${directory}" ]]
  done
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${source_python}" -c \
    'import mujoco, numpy, robosuite; print("LIBERO recovery source imports passed")'
  install -d -m 0755 "${MOUNT_ROOT}"
  staging="$(mktemp -d "${MOUNT_ROOT}/.libero-recovery.XXXXXX")"
  install -d -m 0755 "${staging}/archives"
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
    -czf "${staging}/archives/libero-runtime.tar.gz" \
    -C "${source_root}/miniconda3/envs" libero \
    -C "${source_root}" LIBERO
  gzip -t "${staging}/archives/libero-runtime.tar.gz"
  (
    cd "${staging}/archives"
    sha256sum libero-runtime.tar.gz > libero-runtime.tar.gz.sha256
    sha256sum -c libero-runtime.tar.gz.sha256
  )
  install -m 0755 "${SCRIPT_DIR}/run-benchmark.sh" "${staging}/run-benchmark.sh"
  install -m 0755 "${SCRIPT_DIR}/libero/prepare-runtime.sh" "${staging}/prepare-runtime.sh"
  install -m 0644 "${SCRIPT_DIR}/libero/benchmark-name.txt" "${staging}/benchmark-name.txt"
  printf '%s\n' "${KIT_COMMIT}" > "${staging}/RECOVERY_KIT_COMMIT"
  printf '%s\n' "${source_root}" > "${staging}/RECOVERY_SOURCE_ROOT"
  mv -T -- "${staging}" "${destination}"
  echo "Recovered LIBERO launcher and archive at ${destination} from kit ${KIT_COMMIT}"
}

if [[ -n "${RECOVERY_SOURCE_ROOT}" ]]; then
  recover_libero "${RECOVERY_SOURCE_ROOT}"
  exit 0
fi
SDK_VERSION="$(
  awk -F '=' \
    '/^version[[:space:]]*=/{gsub(/[[:space:]\"]/, "", $2); print $2; exit}' \
    "${SOURCE_KIT_ROOT}/packages/roboscope-kit/pyproject.toml"
)"
if [[ -z "${SDK_VERSION}" ]]; then
  echo "Unable to read SDK version" >&2
  exit 3
fi
CONTROL_SOURCE_SHA256="$(
  python3 "${SCRIPT_DIR}/control-source-sha256.py" "${SOURCE_KIT_ROOT}"
)"
CONTROL_RUNTIME_PARENT="/root/benchmark-runtime/.roboscope-control-runtime"
CONTROL_RUNTIME_ROOT="${CONTROL_RUNTIME_PARENT}/${KIT_COMMIT}/control-runtime"
CONTROL_PYTHON="${CONTROL_RUNTIME_ROOT}/control-python/bin/python"
CONTROL_SOURCE_ROOT="${CONTROL_RUNTIME_ROOT}/packages/roboscope-kit/src"
IMAGE_CONTROL_RUNTIME_ROOT="/opt/roboscope/control-runtime"
IMAGE_CONTROL_PYTHON="${IMAGE_CONTROL_RUNTIME_ROOT}/control-python/bin/python"

verify_control_runtime() {
  [[ "$(<"${CONTROL_RUNTIME_ROOT}/KIT_COMMIT")" == "${KIT_COMMIT}" ]] || return 1
  [[ "$(<"${CONTROL_RUNTIME_ROOT}/SDK_VERSION")" == "${SDK_VERSION}" ]] || return 1
  [[ -x "${CONTROL_PYTHON}" ]] || return 1
  [[ -f "${CONTROL_SOURCE_ROOT}/roboscope_kit/runtime/benchmark/__main__.py" ]] || return 1
  [[ -f "${CONTROL_RUNTIME_ROOT}/benchmarks/robotwin_official/roboscope_robotwin_launcher.py" ]] || return 1
  [[ "$(python3 "${SCRIPT_DIR}/control-source-sha256.py" "${CONTROL_RUNTIME_ROOT}")" == "${CONTROL_SOURCE_SHA256}" ]] || return 1
  env -u PYTHONPATH "${CONTROL_PYTHON}" -I - "${CONTROL_SOURCE_ROOT}" <<'PY'
from pathlib import Path
import sys
import sysconfig

import roboscope_kit.runtime.benchmark
from roboscope_kit.runtime.benchmarks.calvin.adapter import SDK_IMPORT_ROOT as calvin_root
from roboscope_kit.runtime.benchmarks.libero.adapter import SDK_IMPORT_ROOT as libero_root
from roboscope_kit.runtime.benchmarks.libero_plus.adapter import SDK_IMPORT_ROOT as libero_plus_root
from roboscope_kit.runtime.benchmarks.metaworld.adapter import SDK_IMPORT_ROOT as metaworld_root
from roboscope_kit.runtime.benchmarks.robocasa.adapter import SDK_IMPORT_ROOT as robocasa_root

source_root = Path(sys.argv[1]).resolve()
site_packages = Path(sysconfig.get_path("purelib")).resolve()
module_path = Path(roboscope_kit.runtime.benchmark.__file__).resolve()
assert module_path.is_relative_to(source_root)
assert not source_root.is_relative_to(site_packages)
assert {calvin_root, libero_root, libero_plus_root, metaworld_root, robocasa_root} == {source_root}
PY
}

install_control_runtime() {
  local bundle="${ROBOSCOPE_CONTROL_RUNTIME_BUNDLE_DIR:?RoboTwin deployment requires the commit-pinned official image bundle directory}"
  [[ "${bundle}" == /* ]] || {
    echo "Control Runtime bundle must be an absolute directory: ${bundle}" >&2
    exit 3
  }
  (cd "${bundle}" && sha256sum -c payload.sha256 >/dev/null)
  python3 - "${bundle}/manifest.json" "${KIT_COMMIT}" "${SDK_VERSION}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest["evaluation_kit_commit"] != sys.argv[2] or manifest["sdk"]["version"] != sys.argv[3]:
    raise SystemExit("Control Runtime bundle differs from the requested RoboScope Kit build")
PY
  if [[ -e "${CONTROL_RUNTIME_ROOT}" ]]; then
    if ! verify_control_runtime; then
      echo "Existing control runtime is incomplete: ${CONTROL_RUNTIME_ROOT}" >&2
      exit 4
    fi
    return
  fi
  "${SCRIPT_DIR}/install-image-control-runtime.sh" \
    --source-root "${bundle}/control-runtime" \
    --artifacts-dir "${bundle}/control-runtime-artifacts" \
    --target-root "${CONTROL_RUNTIME_ROOT}" \
    --kit-commit "${KIT_COMMIT}" \
    --sdk-version "${SDK_VERSION}"
  # The source checksum covers every RoboTwin preparation entrypoint. Ship the
  # same set so verification compares the installed sources with the checkout.
  install -m 0755 \
    "${SOURCE_KIT_ROOT}/benchmarks/robotwin_official/"*.py \
    "${CONTROL_RUNTIME_ROOT}/benchmarks/robotwin_official/"
  verify_control_runtime
}

if [[ "${#REQUESTED_RUNTIMES[@]}" -eq 0 ]]; then
  install -m 0755 \
    "${SCRIPT_DIR}/prepare-archived-runtime.sh" \
    "${MOUNT_ROOT}/prepare-archived-runtime.sh"
fi

# The local Control Runtime is consumed only by the RoboTwin preparation below.
# Everything else runs on the image-local Control Runtime, so a subset deployment
# without RoboTwin does not require an installation bundle.
if runtime_selected robotwin; then
  install_control_runtime
else
  CONTROL_PYTHON="${IMAGE_CONTROL_PYTHON}"
fi

install_runtime() {
  local source_name="$1"
  local destination_name="$2"
  local destination="${MOUNT_ROOT}/${destination_name}"
  runtime_selected "${destination_name}" || return 0
  install -d -m 0755 "${destination}"
  install -m 0755 "${SCRIPT_DIR}/run-benchmark.sh" "${destination}/run-benchmark.sh"
  install -m 0644 "${SCRIPT_DIR}/${source_name}/benchmark-name.txt" "${destination}/benchmark-name.txt"
  install -m 0644 "${SCRIPT_DIR}/${source_name}/benchmark-image.json" "${destination}/benchmark-image.json"
  "${CONTROL_PYTHON}" -I - \
    "${destination}/benchmark-image.json" \
    "${KIT_COMMIT}" \
    "${SDK_VERSION}" \
    "${IMAGE_CONTROL_RUNTIME_ROOT}" \
    "${IMAGE_CONTROL_PYTHON}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text(encoding="utf-8")
content = content.replace("__EVALUATION_KIT_COMMIT__", sys.argv[2])
content = content.replace("__SDK_VERSION__", sys.argv[3])
content = content.replace("__KIT_ROOT__", sys.argv[4])
content = content.replace("__CONTROL_PYTHON__", sys.argv[5])
path.write_text(content, encoding="utf-8")
PY
  if [[ -f "${SCRIPT_DIR}/${source_name}/prepare-runtime.sh" ]]; then
    if [[ -L "${destination}/runtime/root" ]]; then
      install -m 0755 \
        "${SCRIPT_DIR}/${source_name}/prepare-runtime.sh" \
        "${destination}/prepare-runtime-inner.sh"
      install -m 0755 \
        "${SCRIPT_DIR}/prepare-archived-runtime-wrapper.sh" \
        "${destination}/prepare-runtime.sh"
    else
      install -m 0755 \
        "${SCRIPT_DIR}/${source_name}/prepare-runtime.sh" \
        "${destination}/prepare-runtime.sh"
    fi
  fi
}

install_runtime libero libero
install_runtime libero_plus liberoplus
install_runtime calvin calvin
install_runtime robotwin robotwin

if runtime_selected robotwin; then
  while IFS= read -r suite; do
    env -u PYTHONPATH ROBOSCOPE_CONTROL_PYTHON="${CONTROL_PYTHON}" \
      "${MOUNT_ROOT}/robotwin/prepare-runtime.sh" \
        "/root/benchmark-runtime/robotwin" \
        "${suite}" \
        0
  done < <("${CONTROL_PYTHON}" -I - "${MOUNT_ROOT}/robotwin/benchmark-image.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
for suite in manifest["supported_suites"]["robotwin"]:
    print(suite)
PY
  )
fi

install_runtime robocasa robocasa
install_runtime metaworld metaworld
