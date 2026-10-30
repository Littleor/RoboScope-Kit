#!/usr/bin/env bash
set -Eeuo pipefail

KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_DIR="${KIT_ROOT}/packages/roboscope-kit"
SDK_DIST_DIR="${SDK_DIST_DIR:-${KIT_ROOT}/sdk-dist}"

# Exercise normal PEP 517 builds and wheel installation on every interpreter.
# No source-tree PYTHONPATH may conceal an incompatible distribution.
unset PYTHONPATH
python -m pip install --no-cache-dir "${SDK_DIST_DIR}"/*.tar.gz pytest matplotlib imageio
python -m pip check
python -m pip install --no-cache-dir --force-reinstall --no-deps "${SDK_DIST_DIR}"/*.whl
python - <<'PY'
import importlib
import pkgutil
from importlib.metadata import version

import roboscope_kit as sdk

modules = [
    module.name for module in pkgutil.walk_packages(sdk.__path__, sdk.__name__ + ".")
    if not module.name.endswith(".__main__")
]
for name in modules:
    importlib.import_module(name)
print("SDK {}: imported {} installed modules from {}".format(
    version("roboscope-kit"), len(modules), sdk.__file__
))
PY
python -m compileall -q "$(python -c 'import roboscope_kit as s; print(s.__path__[0])')"

# Only the repository's benchmark assets are added; SDK imports use the wheel.
cd "${SDK_DIR}"
PYTHONPATH="${KIT_ROOT}" python -m pytest \
  --confcutdir="${KIT_ROOT}/tests/async_inference" \
  tests "${KIT_ROOT}/tests/async_inference" -q
