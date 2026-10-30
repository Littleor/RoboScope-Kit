#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_ROOT="${1:?local runtime root is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
INNER_PREPARE="${SCRIPT_DIR}/prepare-runtime-inner.sh"

"${SCRIPT_DIR}/../prepare-archived-runtime.sh" "${SCRIPT_DIR}" "${LOCAL_ROOT}"
[[ -x "${INNER_PREPARE}" ]] || {
  echo "Archived runtime inner preparation is missing: ${INNER_PREPARE}" >&2
  exit 30
}
exec "${INNER_PREPARE}" "$@"
