#!/usr/bin/env bash
set -Eeuo pipefail

ARCHIVE="/root/autodl-fs/benchmarks/libero/archives/runtime-root.tar"
WORKDIR="${ROBOSCOPE_LIBERO_PLATFORM_WORKDIR:-/root/autodl-tmp/roboscope-libero-platform-runtime}"
CHECKSUM_FILE="${WORKDIR}/.archive-sha256"
ENTRYPOINT="${WORKDIR}/evaluation-kit/benchmarks/libero_official/startbenchmark.sh"
RESET_ENTRYPOINT="${WORKDIR}/evaluation-kit/benchmarks/libero_official/resetenv.sh"

archive_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${ARCHIVE}" | awk '{print $1}'
    return
  fi
  shasum -a 256 "${ARCHIVE}" | awk '{print $1}'
}

EXPECTED_SHA256="$(archive_sha256)"
INSTALLED_SHA256="$(cat "${CHECKSUM_FILE}" 2>/dev/null || true)"
if [[ ! -x "${ENTRYPOINT}" || "${INSTALLED_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  rm -rf -- "${WORKDIR}"
  mkdir -p "${WORKDIR}"
  tar -xzf "${ARCHIVE}" --strip-components=1 -C "${WORKDIR}"
  printf '%s\n' "${EXPECTED_SHA256}" > "${CHECKSUM_FILE}"
fi

if [[ "${1:-}" == "--reset" ]]; then
  shift
  exec "${RESET_ENTRYPOINT}" "$@"
fi

exec "${ENTRYPOINT}" "$@"
