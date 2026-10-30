#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_NAME="$(<"${SCRIPT_DIR}/benchmark-name.txt")"
IMAGE_CONTROL_RUNTIME_ROOT="/opt/roboscope/control-runtime"
IMAGE_CONTROL_PYTHON="${IMAGE_CONTROL_RUNTIME_ROOT}/control-python/bin/python"
IMAGE_CONTROL_SDK_ENTRY="${IMAGE_CONTROL_RUNTIME_ROOT}/packages/roboscope-kit/src/roboscope_kit/runtime/benchmark/__main__.py"
LOCAL_ROOT="/root/benchmark-runtime/${BENCHMARK_NAME}"
IMAGE_MANIFEST="${LOCAL_ROOT}/benchmark-image.json"

write_contract() {
  local checkpoint_progress="$1"
  local suite_progress="$2"
  local suite="$3"
  local status="$4"
  mkdir -p /root/testresult /root/testvideo
  local temporary="/root/testlog.txt.$$.tmp"
  cat > "${temporary}" <<EOF
用户模型权重下载进度 ${checkpoint_progress}%
测试套件下载进度 ${suite_progress}%
${BENCHMARK_NAME} ${suite} 0 0 0 ${status}
EOF
  mv "${temporary}" /root/testlog.txt
}

SUITE="none"
on_error() {
  local exit_code="$?"
  trap - ERR
  write_contract 0 0 "${SUITE}" "error:platform_runtime_${exit_code}"
  exit "${exit_code}"
}
trap on_error ERR

fail() {
  local exit_code="$1"
  shift
  echo "$*" >&2
  trap - ERR
  write_contract 0 0 "${SUITE}" "error:platform_runtime_${exit_code}"
  exit "${exit_code}"
}

QUICK_TEST=0
CUDA_MPS=0
EVALUATION_CONFIG=""
SHARD_INDEX=""
SHARD_COUNT=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick-test)
      QUICK_TEST=1
      shift
      ;;
    --cuda-mps)
      CUDA_MPS=1
      shift
      ;;
    --evaluation-config)
      EVALUATION_CONFIG="${2:?--evaluation-config requires a path}"
      shift 2
      ;;
    --shard-index)
      SHARD_INDEX="${2:?--shard-index requires a value}"
      shift 2
      ;;
    --shard-count)
      SHARD_COUNT="${2:?--shard-count requires a value}"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
SUITE="${POSITIONAL[0]:-}"
CHECKPOINT_SOURCE="${POSITIONAL[1]:-}"
if [[ -z "${SUITE}" || -z "${CHECKPOINT_SOURCE}" ]]; then
  fail 2 "Usage: run-benchmark.sh [--quick-test] [--cuda-mps] [--evaluation-config <path>] [--shard-index <i> --shard-count <n>] <suite> <checkpoint_source>"
fi
if [[ -n "${SHARD_INDEX}" || -n "${SHARD_COUNT}" ]]; then
  if [[ -z "${SHARD_INDEX}" || -z "${SHARD_COUNT}" ]]; then
    fail 2 "--shard-index and --shard-count must be given together"
  fi
  if [[ "${QUICK_TEST}" == "1" ]]; then
    fail 2 "--quick-test cannot be combined with sharding"
  fi
fi
if [[ ! -d "${IMAGE_CONTROL_RUNTIME_ROOT}" ]]; then
  fail 5 "Image Control Runtime is missing: ${IMAGE_CONTROL_RUNTIME_ROOT}，请从最新官方基础镜像重建"
fi
if [[ ! -f "${IMAGE_CONTROL_RUNTIME_ROOT}/KIT_COMMIT" || ! -f "${IMAGE_CONTROL_RUNTIME_ROOT}/SDK_VERSION" ]]; then
  fail 5 "Image Control Runtime metadata is incomplete: ${IMAGE_CONTROL_RUNTIME_ROOT}"
fi
if [[ ! -f "${IMAGE_CONTROL_SDK_ENTRY}" ]]; then
  fail 5 "Image Control SDK is incomplete: ${IMAGE_CONTROL_RUNTIME_ROOT}"
fi
if [[ ! -x "${IMAGE_CONTROL_PYTHON}" ]]; then
  fail 3 "Image Control Python is not executable: ${IMAGE_CONTROL_PYTHON}"
fi
CONTROL_PYTHON="${IMAGE_CONTROL_PYTHON}"

# The image owns its SDK and benchmark contract. Materialize that immutable
# reference locally; a shared-disk installation must not change another
# image's contract or require every saved image to have the same kit commit.
SOURCE_MANIFEST_DIR="${IMAGE_CONTROL_RUNTIME_ROOT}/benchmark-manifests/${BENCHMARK_NAME}"
if [[ ! -f "${SOURCE_MANIFEST_DIR}/benchmark-image.json" ]]; then
  fail 4 "Image benchmark manifest is missing: ${SOURCE_MANIFEST_DIR}/benchmark-image.json"
fi
mkdir -p "${LOCAL_ROOT}"
"${CONTROL_PYTHON}" -I - "${IMAGE_MANIFEST}" "${SOURCE_MANIFEST_DIR}/benchmark-image.json" "${IMAGE_CONTROL_RUNTIME_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

installed_path, reference_path, control_root = sys.argv[1:4]
control_root = Path(control_root)
reference = Path(reference_path).read_text(encoding="utf-8")
replacements = {
    "__EVALUATION_KIT_COMMIT__": (control_root / "KIT_COMMIT").read_text(encoding="utf-8").strip(),
    "__SDK_VERSION__": (control_root / "SDK_VERSION").read_text(encoding="utf-8").strip(),
    "__KIT_ROOT__": str(control_root),
    "__CONTROL_PYTHON__": f"{control_root}/control-python/bin/python",
}
for key, value in replacements.items():
    reference = reference.replace(key, value)
manifest = json.loads(reference)
destination = Path(installed_path)
if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != manifest:
    raise SystemExit(
        f"benchmark-image.json deviates from the image source of truth {reference_path}; "
        "manual in-image edits are forbidden"
    )
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
PY

write_contract 0 1 "${SUITE}" preparing
mkdir -p \
  "${LOCAL_ROOT}" \
  "/root/benchmark-cache/${BENCHMARK_NAME}" \
  "/root/benchmark-home/${BENCHMARK_NAME}" \
  "/root/autodl-tmp/roboscope-${BENCHMARK_NAME}"
if [[ -x "${SCRIPT_DIR}/prepare-runtime.sh" ]]; then
  env -u PYTHONPATH ROBOSCOPE_CONTROL_PYTHON="${CONTROL_PYTHON}" \
    "${SCRIPT_DIR}/prepare-runtime.sh" "${LOCAL_ROOT}" "${SUITE}" "${QUICK_TEST}"
fi
write_contract 0 100 "${SUITE}" downloading

export HOME="/root/benchmark-home/${BENCHMARK_NAME}"
export XDG_CACHE_HOME="/root/benchmark-cache/${BENCHMARK_NAME}/xdg"
export HF_HOME="/root/benchmark-cache/${BENCHMARK_NAME}/huggingface"
export TORCH_HOME="/root/benchmark-cache/${BENCHMARK_NAME}/torch"
export TRITON_CACHE_DIR="/root/benchmark-cache/${BENCHMARK_NAME}/triton"
export NUMBA_CACHE_DIR="/root/benchmark-cache/${BENCHMARK_NAME}/numba"
export MPLCONFIGDIR="/root/benchmark-cache/${BENCHMARK_NAME}/matplotlib"
export TMPDIR="/root/autodl-tmp/roboscope-${BENCHMARK_NAME}"
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1

COMMAND=(
  "${CONTROL_PYTHON}"
  -I
  -m roboscope_kit.runtime.benchmark
  --benchmark "${BENCHMARK_NAME}"
  --suite "${SUITE}"
  --checkpoint-source "${CHECKPOINT_SOURCE}"
  --output-dir /root/testresult
  --image-manifest "${IMAGE_MANIFEST}"
  --legacy-contract-root /root
)
if [[ "${QUICK_TEST}" == "1" ]]; then
  COMMAND+=(--quick-test)
fi
if [[ "${CUDA_MPS}" == "1" ]]; then
  COMMAND+=(--cuda-mps)
fi
if [[ -n "${EVALUATION_CONFIG}" ]]; then
  COMMAND+=(--evaluation-config "${EVALUATION_CONFIG}")
fi
if [[ -n "${SHARD_COUNT}" ]]; then
  COMMAND+=(--shard-index "${SHARD_INDEX}" --shard-count "${SHARD_COUNT}")
fi

trap - ERR
exec "${COMMAND[@]}"
