#!/usr/bin/env bash
set -Eeuo pipefail

LOCAL_ROOT="${1:?MetaWorld local runtime root is required}"
ARCHIVE=/root/autodl-fs/benchmarks/metaworld/artifacts/metaworld-env.tar
PYTHON="${LOCAL_ROOT}/metaworld/bin/python"
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "MetaWorld environment archive is missing: ${ARCHIVE}" >&2
  exit 11
fi
if [[ ! -x "${PYTHON}" ]]; then
  tar -xf "${ARCHIVE}" -C "${LOCAL_ROOT}"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "MetaWorld archive did not produce ${PYTHON}" >&2
  exit 12
fi
if ! env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -I -c \
  "import metaworld.evaluation"; then
  echo "MetaWorld runtime cannot import metaworld.evaluation: ${PYTHON}" >&2
  exit 13
fi
