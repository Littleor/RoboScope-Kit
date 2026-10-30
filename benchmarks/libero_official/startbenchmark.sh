#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PLATFORM_ENTRYPOINT="${ROBOSCOPE_PLATFORM_ENTRYPOINT:-}"
if [[ -n "${PLATFORM_ENTRYPOINT}" ]]; then
  if [[ "${PLATFORM_ENTRYPOINT}" != /* ]]; then
    echo "ROBOSCOPE_PLATFORM_ENTRYPOINT must be an absolute path: ${PLATFORM_ENTRYPOINT}" >&2
    exit 3
  fi
  if [[ ! -x "${PLATFORM_ENTRYPOINT}" ]]; then
    echo "Platform Benchmark entrypoint is not executable: ${PLATFORM_ENTRYPOINT}" >&2
    exit 3
  fi
  exec "${PLATFORM_ENTRYPOINT}" "$@"
fi

DEFAULT_RUNNER="${SCRIPT_DIR}/roboscope_libero_runner.py"
if [[ "${SCRIPT_DIR}" == "/root" ]]; then
  DEFAULT_RUNNER="/root/roboscope_libero_runner.py"
else
  export ROBOSCOPE_BASE_ASSET_MAP="${ROBOSCOPE_BASE_ASSET_MAP:-${SCRIPT_DIR}/base_assets.json}"
fi
RUNNER="${ROBOSCOPE_LIBERO_RUNNER:-${DEFAULT_RUNNER}}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/miniconda3/envs/libero/bin/python}"
BENCHMARK_RUNTIME="${ROBOSCOPE_BENCHMARK_RUNTIME:-starvla}"

if [[ "${BENCHMARK_RUNTIME}" != "starvla" && "${BENCHMARK_RUNTIME}" != "policy" ]]; then
  echo "Unsupported benchmark runtime: ${BENCHMARK_RUNTIME}" >&2
  exit 7
fi

if [[ "${1:-}" == "--self-test" ]]; then
  SELF_TEST_PYTHON="${ROBOSCOPE_SELF_TEST_PYTHON:-}"
  if [[ -z "${SELF_TEST_PYTHON}" && "${SCRIPT_DIR}" == "/root" ]]; then
    SELF_TEST_PYTHON="${LIBERO_PYTHON}"
  elif [[ -z "${SELF_TEST_PYTHON}" ]]; then
    SELF_TEST_PYTHON="${SOURCE_KIT_ROOT}/packages/roboscope-kit/.venv/bin/python"
  fi
  SELF_TEST_RUNNER="${ROBOSCOPE_LIBERO_RUNNER:-${SCRIPT_DIR}/roboscope_libero_runner.py}"
  SELF_TEST_KIT_ROOT="${ROBOSCOPE_KIT_ROOT:-}"
  if [[ -z "${SELF_TEST_KIT_ROOT}" && "${SCRIPT_DIR}" == "/root" ]]; then
    SELF_TEST_KIT_ROOT="/root/roboscope-kit"
  elif [[ -z "${SELF_TEST_KIT_ROOT}" ]]; then
    SELF_TEST_KIT_ROOT="${SOURCE_KIT_ROOT}"
  fi
  if [[ ! -x "${SELF_TEST_PYTHON}" ]]; then
    echo "Self-test python not executable: ${SELF_TEST_PYTHON}" >&2
    exit 5
  fi
  ROBOSCOPE_LIBERO_RUNNER="${SELF_TEST_RUNNER}" \
  LIBERO_PYTHON="${SELF_TEST_PYTHON}" \
  ROBOSCOPE_KIT_ROOT="${SELF_TEST_KIT_ROOT}" \
  ROBOSCOPE_BENCHMARK_SELF_TEST=1 \
    exec "${SCRIPT_DIR}/startbenchmark.sh" libero_10 oss://self-test/checkpoint/ 1
fi

if [[ "${1:-}" == "--quick-test" ]]; then
  if [[ -z "${2:-}" || -z "${3:-}" ]]; then
    echo "Usage: /root/startbenchmark.sh --quick-test <suite> <checkpoint_source>" >&2
    exit 2
  fi
  if [[ "${3}" != oss://* ]]; then
    echo "User quick test requires an OSS checkpoint source" >&2
    exit 6
  fi
  required_oss_env=(
    BENCHMARK_CKPT_ENDPOINT
    BENCHMARK_CKPT_ACCESS_KEY_ID
    BENCHMARK_CKPT_ACCESS_KEY_SECRET
  )
  if [[ "${BENCHMARK_RUNTIME}" == "starvla" ]]; then
    required_oss_env+=(
      BENCHMARK_BASE_ENDPOINT
      BENCHMARK_BASE_ACCESS_KEY_ID
      BENCHMARK_BASE_ACCESS_KEY_SECRET
      BENCHMARK_BASE_OSS_URI
    )
  fi
  for name in "${required_oss_env[@]}"; do
    if [[ -z "${!name:-}" ]]; then
      echo "User quick test requires ${name}" >&2
      exit 6
    fi
  done
  BENCHMARK_TASK_IDS=0 \
  BENCHMARK_REPLICAS=1 \
  BENCHMARK_CONCURRENCY=1 \
  BENCHMARK_SAVE_VIDEOS=0 \
  BENCHMARK_QUICK_TEST=1 \
    exec "${SCRIPT_DIR}/startbenchmark.sh" "$2" "$3" 1
fi

if [[ -z "${ROBOSCOPE_KIT_ROOT:-}" ]]; then
  if [[ "${SCRIPT_DIR}" == "/root" ]]; then
    ROBOSCOPE_KIT_ROOT="/root/roboscope-kit"
  else
    ROBOSCOPE_KIT_ROOT="${SOURCE_KIT_ROOT}"
  fi
fi
export ROBOSCOPE_KIT_ROOT
export PYTHONPATH="${ROBOSCOPE_KIT_ROOT}/packages/roboscope-kit/src:${ROBOSCOPE_KIT_ROOT}:${PYTHONPATH:-}"

SUITE="${1:-}"
USER_CHECKPOINT_SOURCE="${2:-}"
EVALUATION_NUM="${3:-50}"

if [[ -z "${SUITE}" || -z "${USER_CHECKPOINT_SOURCE}" ]]; then
  mkdir -p /root/testresult /root/testvideo
  cat > /root/testlog.txt <<LOG
用户模型权重下载进度 0%
测试套件下载进度 0%
LIBERO ${SUITE:-none} 0 0 0 error:missing_args
LOG
  echo "Usage: /root/startbenchmark.sh <suite> <checkpoint_source> [evaluation_num default: 50]" >&2
  exit 2
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO python not executable: ${LIBERO_PYTHON}" >&2
  exit 3
fi

if [[ "${BENCHMARK_RUNTIME}" == "policy" ]]; then
  BENCHMARK_RUNTIME_DIR="${ROBOSCOPE_KIT_ROOT}/packages/roboscope-kit/src/roboscope_kit/runtime/benchmark"
else
  BENCHMARK_RUNTIME_DIR="${ROBOSCOPE_KIT_ROOT}/benchmarks/libero_official/starvla_runtime"
fi

if [[ ! -d "${BENCHMARK_RUNTIME_DIR}" ]]; then
  mkdir -p /root/testresult /root/testvideo
  cat > /root/testlog.txt <<LOG
用户模型权重下载进度 0%
测试套件下载进度 0%
LIBERO ${SUITE} 0 0 0 error:missing_autotune_runtime
LOG
  echo "Missing benchmark runtime at ${BENCHMARK_RUNTIME_DIR}" >&2
  exit 4
fi

exec "${LIBERO_PYTHON}" "${RUNNER}" start \
  --suite "${SUITE}" \
  --checkpoint-source "${USER_CHECKPOINT_SOURCE}" \
  --evaluation-num "${EVALUATION_NUM}"
