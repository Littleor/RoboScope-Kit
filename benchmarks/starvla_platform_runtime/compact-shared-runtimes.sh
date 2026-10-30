#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---preview}"
RUNTIME_NAME="${2:?benchmark is required: liberoplus, robocasa, robotwin}"
MOUNT_ROOT="${3:-/root/autodl-fs/benchmarks}"
ACCEPTANCE_FILE="${4:-}"
case "${RUNTIME_NAME}" in
  liberoplus|robocasa|robotwin) ;;
  shared-control)
    [[ "${MODE}" == --finalize || "${MODE}" == --rollback || "${MODE}" == --preview ]]
    ;;
  *) echo "Select one benchmark: liberoplus, robocasa, robotwin" >&2; exit 2 ;;
esac
RUNTIME_NAMES=("${RUNTIME_NAME}")
JOURNAL="${MOUNT_ROOT}/${RUNTIME_NAME}/.roboscope-compaction"
WRAPPER_SOURCE="${SCRIPT_DIR}/prepare-archived-runtime-wrapper.sh"
HELPER_SOURCE="${SCRIPT_DIR}/prepare-archived-runtime.sh"
ROBOCASA_PREPARE_SOURCE="${SCRIPT_DIR}/robocasa/prepare-runtime.sh"
ARCHIVE_CODEC="zstd"
ARCHIVE_CODEC_VERSION="1.5.7"
ARCHIVE_COMPRESSION_LEVEL="3"
ZSTD_TOOL_ROOT="${MOUNT_ROOT}/.roboscope-runtime-tools/zstd-${ARCHIVE_CODEC_VERSION}"
ZSTD_SOURCE_ROOT="${MOUNT_ROOT}/liberoplus/runtime/root/miniconda3/envs/libero-plus"
ZSTD_BIN_SHA256="08a60ba61031bb1f38070099e77df196b24293de7a2c6517e5f29b183b2299ef"
ZSTD_LIB_SHA256="e32f1e98942e91193d137ae9d460adf8e8cfbf504c8a8aedfb5825576d53a801"
ZLIB_LIB_SHA256="22f1601237b86f0f48ed5b83071d1505167ae2e16365b33b4eed6e96dbf71ab0"

tree_entry_count() {
  find "$1" -xdev -printf . | wc -c | tr -d '[:space:]'
}

tree_regular_file_bytes() {
  find "$1" -xdev -type f -printf '%s\n' |
    awk '{total += $1} END {printf "%.0f\n", total + 0}'
}

runtime_archive_members() {
  local name="$1"
  if [[ "${name}" == "robotwin" ]]; then
    printf '%s\n' root/miniconda3 root/RoboTwin
    return
  fi
  printf '%s\n' root
}

runtime_entry_count() {
  local name="$1"
  local source="$2"
  local members
  readarray -t members < <(runtime_archive_members "${name}")
  if [[ "${members[*]}" == "root" ]]; then
    tree_entry_count "${source}"
    return
  fi
  {
    printf .
    find "${members[@]/#/${source%/root}/}" -xdev -printf .
  } | wc -c | tr -d '[:space:]'
}

runtime_regular_file_bytes() {
  local name="$1"
  local source="$2"
  local members
  readarray -t members < <(runtime_archive_members "${name}")
  if [[ "${members[*]}" == "root" ]]; then
    tree_regular_file_bytes "${source}"
    return
  fi
  find "${members[@]/#/${source%/root}/}" -xdev -type f -printf '%s\n' |
    awk '{total += $1} END {printf "%.0f\n", total + 0}'
}

manifest_value() {
  local manifest="$1"
  local key="$2"
  awk -F= -v key="${key}" '$1 == key {print $2; exit}' "${manifest}"
}

runtime_paths() {
  local name="$1"
  local benchmark_name
  local shared_root="${MOUNT_ROOT}/${name}"
  benchmark_name="$(<"${shared_root}/benchmark-name.txt")"
  printf '%s\n' \
    "${shared_root}/runtime/root" \
    "${shared_root}/archives/runtime-root.tar.zst" \
    "/root/benchmark-runtime/${benchmark_name}/runtime/root"
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
  [[ "$(run_zstd --version)" == *"v${ARCHIVE_CODEC_VERSION}"* ]]
}

install_zstd_tool() {
  local staging
  if [[ -d "${ZSTD_TOOL_ROOT}" ]]; then
    verify_zstd_tool
    return
  fi
  sha256_matches "${ZSTD_BIN_SHA256}" "${ZSTD_SOURCE_ROOT}/bin/zstd"
  sha256_matches "${ZSTD_LIB_SHA256}" "${ZSTD_SOURCE_ROOT}/lib/libzstd.so.1.5.7"
  sha256_matches "${ZLIB_LIB_SHA256}" "${ZSTD_SOURCE_ROOT}/lib/libz.so.1.3.2"
  install -d -m 0755 "${ZSTD_TOOL_ROOT%/*}"
  staging="${ZSTD_TOOL_ROOT}.$$.tmp"
  trap 'rm -rf -- "${staging}"' RETURN
  install -d -m 0755 "${staging}/bin" "${staging}/lib"
  install -m 0755 "${ZSTD_SOURCE_ROOT}/bin/zstd" "${staging}/bin/zstd"
  install -m 0755 \
    "${ZSTD_SOURCE_ROOT}/lib/libzstd.so.1.5.7" \
    "${staging}/lib/libzstd.so.1.5.7"
  install -m 0755 \
    "${ZSTD_SOURCE_ROOT}/lib/libz.so.1.3.2" \
    "${staging}/lib/libz.so.1.3.2"
  ln -s libzstd.so.1.5.7 "${staging}/lib/libzstd.so.1"
  ln -s libz.so.1.3.2 "${staging}/lib/libz.so.1"
  mv -- "${staging}" "${ZSTD_TOOL_ROOT}"
  trap - RETURN
  verify_zstd_tool
}

verify_archive_files() {
  local archive="$1"
  local manifest="${archive}.manifest"
  local actual_sha256
  [[ -f "${archive}" ]]
  [[ -f "${manifest}" ]]
  [[ "$(manifest_value "${manifest}" archive_codec)" == "${ARCHIVE_CODEC}" ]]
  [[ "$(manifest_value "${manifest}" archive_codec_version)" == "${ARCHIVE_CODEC_VERSION}" ]]
  [[ "$(manifest_value "${manifest}" archive_compression_level)" == "${ARCHIVE_COMPRESSION_LEVEL}" ]]
  actual_sha256="$(sha256sum "${archive}" | awk '{print $1}')"
  [[ "${actual_sha256}" == "$(manifest_value "${manifest}" archive_sha256)" ]]
}

verify_archive_against_source() {
  local archive="$1"
  local source="$2"
  local name="$3"
  local manifest="${archive}.manifest"
  verify_archive_files "${archive}"
  [[ "$(runtime_entry_count "${name}" "${source}")" == "$(manifest_value "${manifest}" entry_count)" ]]
  [[ "$(runtime_regular_file_bytes "${name}" "${source}")" == "$(manifest_value "${manifest}" regular_file_bytes)" ]]
  run_zstd --quiet --decompress --stdout "${archive}" |
    tar --compare --file - --directory "${source%/root}"
}

write_manifest() {
  local archive="$1"
  local source="$2"
  local name="$3"
  local temporary="${archive}.manifest.$$.tmp"
  {
    printf 'archive_sha256=%s\n' "$(sha256sum "${archive}" | awk '{print $1}')"
    printf 'archive_codec=%s\n' "${ARCHIVE_CODEC}"
    printf 'archive_codec_version=%s\n' "${ARCHIVE_CODEC_VERSION}"
    printf 'archive_compression_level=%s\n' "${ARCHIVE_COMPRESSION_LEVEL}"
    printf 'entry_count=%s\n' "$(runtime_entry_count "${name}" "${source}")"
    printf 'regular_file_bytes=%s\n' "$(runtime_regular_file_bytes "${name}" "${source}")"
  } > "${temporary}"
  mv -- "${temporary}" "${archive}.manifest"
}

preview_runtime() {
  local name="$1"
  local paths
  readarray -t paths < <(runtime_paths "${name}")
  if [[ -L "${paths[0]}" ]]; then
    printf '%s compact target=%s archive=%s\n' "${name}" "$(readlink "${paths[0]}")" "${paths[1]}"
    return
  fi
  [[ -d "${paths[0]}" ]]
  printf '%s expanded entries=%s bytes=%s archive=%s target=%s\n' \
    "${name}" \
    "$(runtime_entry_count "${name}" "${paths[0]}")" \
    "$(runtime_regular_file_bytes "${name}" "${paths[0]}")" \
    "${paths[1]}" \
    "${paths[2]}"
}

prepare_runtime() {
  local name="$1"
  local paths
  local members
  local temporary
  readarray -t paths < <(runtime_paths "${name}")
  readarray -t members < <(runtime_archive_members "${name}")
  [[ -d "${paths[0]}" && ! -L "${paths[0]}" ]]
  install -d -m 0755 "${paths[1]%/*}"
  if [[ -f "${paths[1]}" ]]; then
    if [[ ! -f "${paths[1]}.manifest" ]]; then
      write_manifest "${paths[1]}" "${paths[0]}" "${name}"
    fi
    verify_archive_against_source "${paths[1]}" "${paths[0]}" "${name}"
    return
  fi
  temporary="${paths[1]}.$$.tmp"
  trap 'rm -f -- "${temporary}"' RETURN
  tar --create --file - --directory "${paths[0]%/root}" "${members[@]}" |
    run_zstd \
      --quiet \
      --threads=0 \
      --check \
      -"${ARCHIVE_COMPRESSION_LEVEL}" \
      -o "${temporary}"
  mv -- "${temporary}" "${paths[1]}"
  trap - RETURN
  write_manifest "${paths[1]}" "${paths[0]}" "${name}"
  verify_archive_against_source "${paths[1]}" "${paths[0]}" "${name}"
}

preflight_apply_runtime() {
  local name="$1"
  local expanded_backup
  local paths
  readarray -t paths < <(runtime_paths "${name}")
  expanded_backup="${JOURNAL}/expanded/root"
  verify_archive_files "${paths[1]}"
  if [[ -L "${paths[0]}" ]]; then
    [[ "$(readlink "${paths[0]}")" == "${paths[2]}" ]]
    return
  fi
  if [[ -d "${expanded_backup}" ]]; then
    [[ ! -e "${paths[0]}" ]]
    verify_archive_against_source "${paths[1]}" "${expanded_backup}" "${name}"
    return
  fi
  [[ -d "${paths[0]}" ]]
  verify_archive_against_source "${paths[1]}" "${paths[0]}" "${name}"
}

preflight_runtime_wrapper() {
  local name="$1"
  local inner="${MOUNT_ROOT}/${name}/prepare-runtime-inner.sh"
  local prepare="${MOUNT_ROOT}/${name}/prepare-runtime.sh"
  [[ -x "${WRAPPER_SOURCE}" ]]
  [[ -x "${HELPER_SOURCE}" ]]
  [[ -x "${ROBOCASA_PREPARE_SOURCE}" ]]
  if [[ -e "${inner}" ]]; then
    [[ -x "${inner}" ]]
    if [[ -e "${prepare}" ]]; then
      cmp --silent "${WRAPPER_SOURCE}" "${prepare}"
    fi
    return
  fi
  if [[ -e "${prepare}" ]]; then
    [[ -x "${prepare}" ]]
    if cmp --silent "${WRAPPER_SOURCE}" "${prepare}"; then
      [[ "${name}" == "robocasa" ]]
    fi
    return
  fi
  [[ "${name}" == "robocasa" ]]
}

install_robocasa_inner() {
  local inner="$1"
  local temporary="${inner}.$$.tmp"
  install -m 0755 "${ROBOCASA_PREPARE_SOURCE}" "${temporary}"
  mv -- "${temporary}" "${inner}"
}

install_runtime_helper() {
  local destination="${MOUNT_ROOT}/prepare-archived-runtime.sh"
  local temporary="${destination}.$$.tmp"
  if [[ -e "${destination}" ]]; then
    cmp --silent "${HELPER_SOURCE}" "${destination}"
    return
  fi
  install -m 0755 "${HELPER_SOURCE}" "${temporary}"
  mv -- "${temporary}" "${destination}"
}

install_runtime_wrapper() {
  local name="$1"
  local inner="${MOUNT_ROOT}/${name}/prepare-runtime-inner.sh"
  local prepare="${MOUNT_ROOT}/${name}/prepare-runtime.sh"
  local temporary="${prepare}.$$.tmp"
  if [[ -e "${prepare}" ]] && cmp --silent "${WRAPPER_SOURCE}" "${prepare}"; then
    if [[ ! -e "${inner}" ]]; then
      [[ "${name}" == "robocasa" ]]
      install_robocasa_inner "${inner}"
    fi
    return
  fi
  install -m 0755 "${WRAPPER_SOURCE}" "${temporary}"
  if [[ -e "${prepare}" && ! -e "${inner}" ]]; then
    mv -- "${prepare}" "${inner}"
    trap 'mv -- "${inner}" "${prepare}"; rm -f -- "${temporary}"' RETURN
  elif [[ ! -e "${inner}" ]]; then
    [[ "${name}" == "robocasa" ]]
    install_robocasa_inner "${inner}"
    trap 'rm -f -- "${inner}" "${temporary}"' RETURN
  fi
  mv -- "${temporary}" "${prepare}"
  trap - RETURN
}

apply_runtime() {
  local name="$1"
  local expanded_backup
  local paths
  local temporary_link
  readarray -t paths < <(runtime_paths "${name}")
  expanded_backup="${JOURNAL}/expanded/root"
  if [[ -L "${paths[0]}" ]]; then
    return
  fi
  temporary_link="${paths[0]}.$$.tmp"
  ln -s "${paths[2]}" "${temporary_link}"
  if [[ ! -d "${expanded_backup}" ]]; then
    mkdir -p -- "${JOURNAL}/expanded"
    mv -- "${paths[0]}" "${expanded_backup}"
  fi
  trap 'mv -- "${expanded_backup}" "${paths[0]}"; rm -f -- "${temporary_link}"' RETURN
  mv -- "${temporary_link}" "${paths[0]}"
  trap - RETURN
}

write_phase() {
  printf '%s\n' "$1" > "${JOURNAL}/phase.tmp"
  mv -- "${JOURNAL}/phase.tmp" "${JOURNAL}/phase"
}

save_original_contract() {
  local shared_root="${MOUNT_ROOT}/${RUNTIME_NAME}"
  local staging="${JOURNAL}.$$.tmp"
  if [[ -d "${JOURNAL}" ]]; then
    verify_original_contract
    return
  fi
  [[ -d "${shared_root}/runtime/root" && ! -L "${shared_root}/runtime/root" ]]
  [[ ! -e "${shared_root}/prepare-runtime-inner.sh" ]]
  mkdir -- "${staging}"
  if [[ -e "${shared_root}/prepare-runtime.sh" ]]; then
    cp -p -- "${shared_root}/prepare-runtime.sh" "${staging}/prepare-runtime.original"
    printf 'present\n' > "${staging}/prepare-presence"
  else
    printf 'absent\n' > "${staging}/prepare-presence"
  fi
  (
    cd "${shared_root}"
    sha256sum benchmark-image.json run-benchmark.sh benchmark-name.txt
  ) > "${staging}/contract.sha256"
  (
    cd "${staging}"
    sha256sum prepare-presence contract.sha256
    if [[ -f prepare-runtime.original ]]; then
      sha256sum prepare-runtime.original
    fi
  ) > "${staging}/original.sha256"
  git -C "${SCRIPT_DIR}/../.." rev-parse HEAD > "${staging}/prepared-commit"
  mv -- "${staging}" "${JOURNAL}"
  write_phase preparing
}

verify_original_contract() {
  (cd "${JOURNAL}" && sha256sum --check original.sha256)
  (cd "${MOUNT_ROOT}/${RUNTIME_NAME}" && sha256sum --check "${JOURNAL}/contract.sha256")
}

recovery_archive() {
  if [[ "${RUNTIME_NAME}" == robotwin ]]; then
    printf '%s\n' "${JOURNAL}/recovery-root.tar.zst"
    return
  fi
  printf '%s\n' "${MOUNT_ROOT}/${RUNTIME_NAME}/archives/runtime-root.tar.zst"
}

prepare_recovery_archive() {
  local archive
  local source="${MOUNT_ROOT}/${RUNTIME_NAME}/runtime/root"
  archive="$(recovery_archive)"
  if [[ ! -f "${archive}" ]]; then
    rm -f -- "${archive}.tmp"
    tar --create --file - --directory "${source%/root}" root |
      run_zstd --quiet --threads=0 --check -"${ARCHIVE_COMPRESSION_LEVEL}" -o "${archive}.tmp"
    mv -- "${archive}.tmp" "${archive}"
  fi
  if [[ ! -f "${archive}.manifest" ]]; then
    write_manifest "${archive}" "${source}" liberoplus
  fi
  verify_archive_against_source "${archive}" "${source}" liberoplus
}

verify_expanded_preparation() {
  local prepare="${MOUNT_ROOT}/${RUNTIME_NAME}/prepare-runtime.sh"
  if [[ "$(<"${JOURNAL}/prepare-presence")" == present ]]; then
    cmp --silent "${JOURNAL}/prepare-runtime.original" "${prepare}"
    return
  fi
  [[ ! -e "${prepare}" ]]
}

verify_compact_preparation() {
  local shared_root="${MOUNT_ROOT}/${RUNTIME_NAME}"
  cmp --silent "${WRAPPER_SOURCE}" "${shared_root}/prepare-runtime.sh"
  cmp --silent "${HELPER_SOURCE}" "${MOUNT_ROOT}/prepare-archived-runtime.sh"
  if [[ "$(<"${JOURNAL}/prepare-presence")" == present ]]; then
    cmp --silent "${JOURNAL}/prepare-runtime.original" "${shared_root}/prepare-runtime-inner.sh"
    return
  fi
  [[ "${RUNTIME_NAME}" == robocasa ]]
  cmp --silent "${ROBOCASA_PREPARE_SOURCE}" "${shared_root}/prepare-runtime-inner.sh"
}

verify_acceptance() {
  local archive="${MOUNT_ROOT}/${RUNTIME_NAME}/archives/runtime-root.tar.zst"
  local evidence_file
  [[ -f "${ACCEPTANCE_FILE}" ]]
  [[ "$(manifest_value "${ACCEPTANCE_FILE}" benchmark)" == "${RUNTIME_NAME}" ]]
  [[ "$(manifest_value "${ACCEPTANCE_FILE}" archive_sha256)" == "$(manifest_value "${archive}.manifest" archive_sha256)" ]]
  [[ "$(manifest_value "${ACCEPTANCE_FILE}" end_to_end_status)" == passed ]]
  [[ "$(manifest_value "${ACCEPTANCE_FILE}" rollback_reapply_status)" == passed ]]
  [[ "$(manifest_value "${ACCEPTANCE_FILE}" cutover_id)" == "$(<"${JOURNAL}/cutover-id")" ]]
  [[ -n "$(manifest_value "${ACCEPTANCE_FILE}" task_id)" ]]
  evidence_file="$(manifest_value "${ACCEPTANCE_FILE}" evidence_file)"
  [[ "${evidence_file}" == /* && -f "${evidence_file}" ]]
  sha256_matches "$(manifest_value "${ACCEPTANCE_FILE}" evidence_sha256)" "${evidence_file}"
}

finalize_runtime() {
  local source="${MOUNT_ROOT}/${RUNTIME_NAME}/runtime/root"
  local backup="${JOURNAL}/expanded/root"
  local archive
  archive="$(recovery_archive)"
  verify_original_contract
  verify_compact_preparation
  verify_acceptance
  [[ -L "${source}" ]]
  [[ "$(readlink "${source}")" == "$(runtime_paths "${RUNTIME_NAME}" | tail -n 1)" ]]
  case "$(<"${JOURNAL}/phase")" in
    applied) [[ -d "${backup}" ]] ;;
    reclaiming|finalized) ;;
    *) echo "Finalize requires an applied and accepted runtime" >&2; exit 4 ;;
  esac
  verify_archive_files "${archive}"
  if [[ -d "${backup}" ]]; then
    if [[ "$(<"${JOURNAL}/phase")" != reclaiming ]]; then
      verify_archive_against_source "${archive}" "${backup}" liberoplus
    fi
    cp -- "${ACCEPTANCE_FILE}" "${JOURNAL}/acceptance"
    write_phase reclaiming
    rm -rf -- "${backup}"
  fi
  write_phase finalized
}

restore_archive() {
  local archive="$1"
  local destination="$2"
  run_zstd --quiet --decompress --stdout "${archive}" | tar -xf - -C "${destination}"
  # AutoDL FS can replace tar's timestamps when buffered file writes finish.
  # Flush those writes before restoring archive timestamps through path operations.
  sync -f "${destination}"
  run_zstd --quiet --decompress --stdout "${archive}" |
    python3 -c '
import os
import sys
import tarfile

os.chdir(sys.argv[1])
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
    for member in archive:
        os.utime(member.name, (member.mtime, member.mtime), follow_symlinks=False)
' "${destination}"
}

restore_expanded_root() {
  local source="${MOUNT_ROOT}/${RUNTIME_NAME}/runtime/root"
  local backup="${JOURNAL}/expanded/root"
  local staging="${JOURNAL}/restore"
  local archive
  archive="$(recovery_archive)"
  if [[ -L "${source}" ]]; then
    [[ "$(readlink "${source}")" == "$(runtime_paths "${RUNTIME_NAME}" | tail -n 1)" ]]
  fi
  if [[ -d "${source}" && ! -L "${source}" ]]; then
    return
  fi
  if [[ ! -d "${backup}" ]]; then
    verify_archive_files "${archive}"
    rm -rf -- "${staging}"
    verify_restore_capacity "${archive}"
    mkdir -- "${staging}"
    restore_archive "${archive}" "${staging}"
    verify_archive_against_source "${archive}" "${staging}/root" liberoplus
    mkdir -p -- "${JOURNAL}/expanded"
    mv -- "${staging}/root" "${backup}"
    rmdir -- "${staging}"
  fi
  [[ ! -e "${source}" || -L "${source}" ]]
  if [[ -L "${source}" ]]; then
    rm -- "${source}"
  fi
  mv -- "${backup}" "${source}"
}

rollback_runtime() {
  local shared_root="${MOUNT_ROOT}/${RUNTIME_NAME}"
  verify_original_contract
  verify_archive_files "$(recovery_archive)"
  if [[ "$(<"${JOURNAL}/phase")" == reclaiming ]]; then
    rm -rf -- "${JOURNAL}/expanded/root"
  fi
  write_phase restoring
  restore_expanded_root
  if [[ "$(<"${JOURNAL}/prepare-presence")" == present ]]; then
    cp -p -- "${JOURNAL}/prepare-runtime.original" "${shared_root}/prepare-runtime.sh.tmp"
    mv -- "${shared_root}/prepare-runtime.sh.tmp" "${shared_root}/prepare-runtime.sh"
  else
    rm -f -- "${shared_root}/prepare-runtime.sh"
  fi
  rm -f -- "${shared_root}/prepare-runtime-inner.sh"
  verify_expanded_preparation
  verify_original_contract
  write_phase rolled-back
}

verify_restore_capacity() {
  local archive="$1"
  local expected_entries
  local required_bytes
  local available_bytes
  local available_inodes
  expected_entries="$(manifest_value "${archive}.manifest" entry_count)"
  required_bytes=$(( $(manifest_value "${archive}.manifest" regular_file_bytes) + expected_entries * 4096 + 1073741824 ))
  available_bytes="$(df -PB1 "${MOUNT_ROOT}" | awk 'NR == 2 {print $4}')"
  available_inodes="$(df -Pi "${MOUNT_ROOT}" | awk 'NR == 2 {print $4}')"
  [[ "${available_bytes}" -gt "${required_bytes}" ]]
  [[ "${available_inodes}" -gt "$((expected_entries + 100))" ]]
}

acquire_rollout_lock() {
  local lock="${MOUNT_ROOT}/.roboscope-compaction-lock"
  mkdir -- "${lock}"
  trap 'rm -f -- "${MOUNT_ROOT}/.roboscope-compaction-lock/owner"; rmdir -- "${MOUNT_ROOT}/.roboscope-compaction-lock"' EXIT
  printf 'host=%s\npid=%s\nbenchmark=%s\nmode=%s\n' \
    "$(hostname)" "$$" "${RUNTIME_NAME}" "${MODE}" > "${lock}/owner"
}

verify_control_recovery() {
  local archive="${MOUNT_ROOT}/.roboscope-control-recovery/control-runtime.tar.zst"
  (cd "${archive%/*}" && sha256sum --check archive.sha256)
  run_zstd --quiet --test "${archive}"
}

finalize_shared_control() {
  local recovery="${MOUNT_ROOT}/.roboscope-control-recovery"
  local source="${MOUNT_ROOT}/.roboscope-control-runtime"
  local archive="${recovery}/control-runtime.tar.zst"
  local name
  for name in liberoplus robocasa robotwin; do
    [[ "$(<"${MOUNT_ROOT}/${name}/.roboscope-compaction/phase")" == finalized ]]
    [[ -s "${MOUNT_ROOT}/${name}/.roboscope-compaction/acceptance" ]]
  done
  install -d -m 0755 "${recovery}"
  if [[ -f "${recovery}/finalized" ]]; then
    [[ ! -e "${source}" ]]
    verify_control_recovery
    return
  fi
  if [[ ! -f "${archive}" ]]; then
    [[ -d "${source}" && ! -L "${source}" ]]
    rm -f -- "${archive}.tmp"
    tar --create --file - --directory "${MOUNT_ROOT}" .roboscope-control-runtime |
      run_zstd --quiet --threads=0 --check -"${ARCHIVE_COMPRESSION_LEVEL}" -o "${archive}.tmp"
    mv -- "${archive}.tmp" "${archive}"
  fi
  if [[ ! -f "${recovery}/archive.sha256" ]]; then
    (cd "${recovery}" && sha256sum control-runtime.tar.zst) > "${recovery}/archive.sha256"
  fi
  verify_control_recovery
  if [[ ! -f "${recovery}/reclaiming" ]]; then
    run_zstd --quiet --decompress --stdout "${archive}" |
      tar --compare --file - --directory "${MOUNT_ROOT}"
    printf '%s\n' "$(tree_entry_count "${source}")" > "${recovery}/entry-count"
    printf '%s\n' "$(tree_regular_file_bytes "${source}")" > "${recovery}/regular-bytes"
    touch "${recovery}/reclaiming"
  fi
  rm -rf -- "${source}"
  mv -- "${recovery}/reclaiming" "${recovery}/finalized"
}

rollback_shared_control() {
  local recovery="${MOUNT_ROOT}/.roboscope-control-recovery"
  local source="${MOUNT_ROOT}/.roboscope-control-runtime"
  local staging="${recovery}/restore"
  local archive="${recovery}/control-runtime.tar.zst"
  verify_control_recovery
  if [[ -d "${source}" && ! -e "${recovery}/reclaiming" ]]; then
    run_zstd --quiet --decompress --stdout "${archive}" | tar --compare --file - --directory "${MOUNT_ROOT}"
    rm -f -- "${recovery}/finalized"
    return
  fi
  rm -rf -- "${staging}"
  mkdir -- "${staging}"
  restore_archive "${archive}" "${staging}"
  run_zstd --quiet --decompress --stdout "${archive}" | tar --compare --file - --directory "${staging}"
  if [[ -f "${recovery}/reclaiming" ]]; then
    rm -rf -- "${source}"
  fi
  [[ ! -e "${source}" ]]
  mv -- "${staging}/.roboscope-control-runtime" "${source}"
  rmdir -- "${staging}"
  rm -f -- "${recovery}/reclaiming" "${recovery}/finalized"
}

verify_checkout() {
  local commit
  commit="$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD)"
  [[ "${commit}" =~ ^[0-9a-f]{40}$ ]]
  [[ -z "$(git -C "${SCRIPT_DIR}/../.." status --short --untracked-files=no)" ]]
}

verify_mount_root() {
  [[ "${MOUNT_ROOT}" == /* ]]
  [[ "${MOUNT_ROOT}" != "/" ]]
  if [[ "${RUNTIME_NAME}" == shared-control ]]; then
    [[ -d "${MOUNT_ROOT}/liberoplus/.roboscope-compaction" ]]
    return
  fi
  local name
  for name in "${RUNTIME_NAMES[@]}"; do
    [[ -f "${MOUNT_ROOT}/${name}/benchmark-name.txt" ]]
  done
}

verify_prepare_capacity() {
  local available_bytes
  local name
  local paths
  local required_bytes=0
  for name in "${RUNTIME_NAMES[@]}"; do
    readarray -t paths < <(runtime_paths "${name}")
    if [[ -f "${paths[1]}" ]]; then
      continue
    fi
    required_bytes=$((
      required_bytes +
      $(runtime_regular_file_bytes "${name}" "${paths[0]}") +
      $(runtime_entry_count "${name}" "${paths[0]}") * 1024
    ))
  done
  if [[ "${RUNTIME_NAME}" == robotwin && ! -f "$(recovery_archive)" ]]; then
    required_bytes=$((required_bytes + $(tree_regular_file_bytes "${MOUNT_ROOT}/robotwin/runtime/root") + $(tree_entry_count "${MOUNT_ROOT}/robotwin/runtime/root") * 1024))
  fi
  available_bytes="$(df -PB1 "${MOUNT_ROOT}" | awk 'NR == 2 {print $4}')"
  [[ "${available_bytes}" -gt "${required_bytes}" ]] || {
    echo "Shared disk requires ${required_bytes} bytes for archives; ${available_bytes} bytes are available" >&2
    exit 3
  }
}

verify_checkout
verify_mount_root
case "${MODE}" in
  --preview) ;;
  --prepare|--apply|--rollback|--finalize) acquire_rollout_lock ;;
  *) echo "Select --preview, --prepare, --apply, --rollback, or --finalize" >&2; exit 2 ;;
esac
if [[ "${RUNTIME_NAME}" == shared-control ]]; then
  verify_zstd_tool
  case "${MODE}" in
    --preview) df -i "${MOUNT_ROOT}" ;;
    --finalize) finalize_shared_control ;;
    --rollback) rollback_shared_control ;;
  esac
  exit 0
fi
case "${MODE}" in
  --preview)
    for name in "${RUNTIME_NAMES[@]}"; do
      preview_runtime "${name}"
    done
    if [[ -d "${MOUNT_ROOT}/.roboscope-control-runtime" ]]; then
      printf 'shared control runtime entries=%s\n' \
        "$(tree_entry_count "${MOUNT_ROOT}/.roboscope-control-runtime")"
    fi
    if [[ -f "${JOURNAL}/phase" ]]; then
      printf 'phase=%s\n' "$(<"${JOURNAL}/phase")"
    fi
    ;;
  --prepare)
    save_original_contract
    verify_expanded_preparation
    verify_prepare_capacity
    install_zstd_tool
    for name in "${RUNTIME_NAMES[@]}"; do
      prepare_runtime "${name}"
    done
    prepare_recovery_archive
    write_phase prepared
    ;;
  --apply)
    verify_zstd_tool
    verify_original_contract
    verify_archive_files "$(recovery_archive)"
    case "$(<"${JOURNAL}/phase")" in
      prepared|rolled-back|applying) ;;
      applied|finalized)
        preflight_apply_runtime "${RUNTIME_NAME}"
        verify_compact_preparation
        preview_runtime "${RUNTIME_NAME}"
        exit 0
        ;;
      *) echo "Apply requires prepared archives or a rolled-back runtime" >&2; exit 4 ;;
    esac
    if [[ "$(<"${JOURNAL}/phase")" == prepared || "$(<"${JOURNAL}/phase")" == rolled-back ]]; then
      verify_expanded_preparation
    fi
    for name in "${RUNTIME_NAMES[@]}"; do
      preflight_apply_runtime "${name}"
      preflight_runtime_wrapper "${name}"
    done
    if [[ "$(<"${JOURNAL}/phase")" != applying ]]; then
      printf '%s-%s-%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "$(hostname)" "$$" > "${JOURNAL}/cutover-id"
      write_phase applying
    fi
    install_runtime_helper
    for name in "${RUNTIME_NAMES[@]}"; do
      install_runtime_wrapper "${name}"
    done
    for name in "${RUNTIME_NAMES[@]}"; do
      apply_runtime "${name}"
    done
    verify_original_contract
    verify_compact_preparation
    write_phase applied
    for name in "${RUNTIME_NAMES[@]}"; do
      preview_runtime "${name}"
    done
    ;;
  --rollback)
    verify_zstd_tool
    rollback_runtime
    preview_runtime "${RUNTIME_NAME}"
    ;;
  --finalize)
    verify_zstd_tool
    finalize_runtime
    preview_runtime "${RUNTIME_NAME}"
    ;;
  *)
    echo "Usage: compact-shared-runtimes.sh MODE BENCHMARK [mount-root] [acceptance-file]" >&2
    exit 2
    ;;
esac
