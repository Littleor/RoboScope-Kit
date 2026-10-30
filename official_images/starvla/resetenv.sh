#!/usr/bin/env bash
set -Eeuo pipefail

PLATFORM_RESET="${ROBOSCOPE_PLATFORM_RESET:-}"
if [[ -n "${PLATFORM_RESET}" ]]; then
  if [[ "${PLATFORM_RESET}" != /* ]]; then
    echo "ROBOSCOPE_PLATFORM_RESET must be an absolute path: ${PLATFORM_RESET}" >&2
    exit 3
  fi
  if [[ ! -x "${PLATFORM_RESET}" ]]; then
    echo "Platform Benchmark reset is not executable: ${PLATFORM_RESET}" >&2
    exit 3
  fi
  exec "${PLATFORM_RESET}" --reset "$@"
fi
