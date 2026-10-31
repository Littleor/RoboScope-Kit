#!/usr/bin/env bash
set -Eeuo pipefail

PLATFORM_ENTRYPOINT="${ROBOSCOPE_PLATFORM_ENTRYPOINT:-}"
if [[ -z "${PLATFORM_ENTRYPOINT}" ]]; then
  echo "ROBOSCOPE_PLATFORM_ENTRYPOINT is required" >&2
  exit 3
fi
if [[ "${PLATFORM_ENTRYPOINT}" != /* ]]; then
  echo "ROBOSCOPE_PLATFORM_ENTRYPOINT must be an absolute path: ${PLATFORM_ENTRYPOINT}" >&2
  exit 3
fi
if [[ ! -x "${PLATFORM_ENTRYPOINT}" ]]; then
  echo "Platform Benchmark entrypoint is not executable: ${PLATFORM_ENTRYPOINT}" >&2
  exit 3
fi
exec "${PLATFORM_ENTRYPOINT}" "$@"
