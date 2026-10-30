#!/usr/bin/env bash
set -Eeuo pipefail

SHARED_BENCHMARK_ROOT="${1:?shared benchmark root is required}"
LOCAL_BENCHMARK_ROOT="${2:?local benchmark root is required}"
ARCHIVE_CODEC="zstd"
ARCHIVE_CODEC_VERSION="1.5.7"
ARCHIVE_COMPRESSION_LEVEL="3"
SHARED_MOUNT_ROOT="${SHARED_BENCHMARK_ROOT%/*}"
ZSTD_TOOL_ROOT="${SHARED_MOUNT_ROOT}/.roboscope-runtime-tools/zstd-${ARCHIVE_CODEC_VERSION}"
ZSTD_BIN_SHA256="08a60ba61031bb1f38070099e77df196b24293de7a2c6517e5f29b183b2299ef"
ZSTD_LIB_SHA256="e32f1e98942e91193d137ae9d460adf8e8cfbf504c8a8aedfb5825576d53a801"
ZLIB_LIB_SHA256="22f1601237b86f0f48ed5b83071d1505167ae2e16365b33b4eed6e96dbf71ab0"
ARCHIVE="${SHARED_BENCHMARK_ROOT}/archives/runtime-root.tar.zst"
MANIFEST="${ARCHIVE}.manifest"
SHARED_RUNTIME_ROOT="${SHARED_BENCHMARK_ROOT}/runtime/root"
LOCAL_RUNTIME_PARENT="${LOCAL_BENCHMARK_ROOT}/runtime"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_PARENT}/root"

manifest_value() {
  local key="$1"
  awk -F= -v key="${key}" '$1 == key {print $2; exit}' "${MANIFEST}"
}

sha256_matches() {
  local expected="$1"
  local path="$2"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
}

run_zstd() {
  env LD_LIBRARY_PATH="${ZSTD_TOOL_ROOT}/lib" \
    "${ZSTD_TOOL_ROOT}/bin/zstd" "$@"
}

verify_zstd_tool() {
  sha256_matches "${ZSTD_BIN_SHA256}" "${ZSTD_TOOL_ROOT}/bin/zstd"
  sha256_matches "${ZSTD_LIB_SHA256}" "${ZSTD_TOOL_ROOT}/lib/libzstd.so.1.5.7"
  sha256_matches "${ZLIB_LIB_SHA256}" "${ZSTD_TOOL_ROOT}/lib/libz.so.1.3.2"
  [[ "$(readlink "${ZSTD_TOOL_ROOT}/lib/libzstd.so.1")" == "libzstd.so.1.5.7" ]]
  [[ "$(readlink "${ZSTD_TOOL_ROOT}/lib/libz.so.1")" == "libz.so.1.3.2" ]]
}

verify_shared_contract() {
  local expected_target="${LOCAL_RUNTIME_ROOT}"
  [[ -f "${ARCHIVE}" ]] || {
    echo "Archived runtime is missing: ${ARCHIVE}" >&2
    exit 21
  }
  [[ -f "${MANIFEST}" ]] || {
    echo "Archived runtime manifest is missing: ${MANIFEST}" >&2
    exit 22
  }
  [[ "$(manifest_value archive_codec)" == "${ARCHIVE_CODEC}" ]]
  [[ "$(manifest_value archive_codec_version)" == "${ARCHIVE_CODEC_VERSION}" ]]
  [[ "$(manifest_value archive_compression_level)" == "${ARCHIVE_COMPRESSION_LEVEL}" ]]
  verify_zstd_tool
  [[ -L "${SHARED_RUNTIME_ROOT}" ]] || {
    echo "Shared runtime root must be a symbolic link: ${SHARED_RUNTIME_ROOT}" >&2
    exit 23
  }
  [[ "$(readlink "${SHARED_RUNTIME_ROOT}")" == "${expected_target}" ]] || {
    echo "Shared runtime root points outside ${expected_target}: ${SHARED_RUNTIME_ROOT}" >&2
    exit 24
  }
}

extract_runtime() {
  local available_bytes
  local expected_bytes
  local expected_entries
  local required_bytes
  local staging="${LOCAL_BENCHMARK_ROOT}/.runtime.$$.tmp"

  expected_bytes="$(manifest_value regular_file_bytes)"
  expected_entries="$(manifest_value entry_count)"
  required_bytes=$((expected_bytes + expected_entries * 4096 + 1073741824))
  available_bytes="$(df -PB1 "${LOCAL_BENCHMARK_ROOT}" | awk 'NR == 2 {print $4}')"
  [[ "${available_bytes}" -gt "${required_bytes}" ]] || {
    echo "Local runtime disk requires ${required_bytes} bytes including inode and staging headroom; ${available_bytes} bytes are available" >&2
    exit 25
  }
  rm -rf -- "${staging}"
  mkdir -p "${staging}"
  trap 'rm -rf -- "${staging}"' EXIT
  run_zstd --quiet --decompress --stdout "${ARCHIVE}" |
    tar -xf - -C "${staging}"
  [[ -d "${staging}/root" ]] || {
    echo "Archived runtime did not produce root/" >&2
    exit 26
  }
  rm -rf -- "${LOCAL_RUNTIME_PARENT}"
  mv -- "${staging}" "${LOCAL_RUNTIME_PARENT}"
  trap - EXIT
}

verify_shared_contract
extract_runtime
[[ -d "${LOCAL_RUNTIME_ROOT}" ]]
