#!/usr/bin/env bash
set -Eeuo pipefail

# Build the RoboTwin 2.0 benchmark runtime under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json          this directory's manifest; runtime paths are relative to it
#     env/                             conda environment (CPython 3.8), runtime.benchmark_python
#     RoboTwin/                        pinned RoboTwin tree with its assets and envs/curobo
#     roboscope_robotwin_launcher.py   runs RoboTwin's evaluator inside env/
#     roboscope_robotwin_seed_scan.py  the scanner whose SHA-256 the seed generations record
#     seed-cache-v2.tar.gz             the frozen seed archive
#     seed-cache/<suite>/<task>.pkl    the archive extracted; used by --quick-test
#     seed-generations/<digest>/       one verified generation per suite; used by formal runs
#
# Usage: build-runtime.sh [--curobo-commit <sha>] [--assets-source <dataset>] <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark robotwin \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Environment:
#   KIT_PYTHON                     Python with roboscope-kit installed (default python3); builds the
#                                  seed generations and runs the self-test. The RoboTwin environment
#                                  does not need the kit.
#   ROBOSCOPE_ROBOTWIN_ASSETS_DIR  optional directory holding background_texture.zip,
#                                  embodiments.zip and objects.zip; they are verified, not refetched.
#   ROBOSCOPE_BUILD_MAX_JOBS       parallel CUDA compile jobs (default 1).
#
# Re-running on an existing <install-dir> reuses the RoboTwin tree (verified against the
# runtime commit), the environment and the cuRobo build, and re-verifies the seed inputs.
#
# Pin files committed next to this script:
#   conda-specs.txt           exact name-version-build strings recovered from the
#                             original runtime's conda-meta directory
#   runtime-requirements.txt  exact pip versions recovered from the original
#                             runtime's top-level *.dist-info directories
#
# The RoboTwin tree is pinned to the base commit recorded in benchmark-manifest.json,
# and the deterministic annotation patch below must reproduce the exact runtime
# commit the benchmark protocol names. The seed generation digests depend on it.
#
# Three inputs have no surviving record of their exact version and are resolved
# from the pinned RoboTwin tree's own committed install script, script/_install.sh
# at the base commit:
#   cuRobo        cloned into envs/curobo from https://github.com/NVlabs/curobo
#                 at tag v0.7.8 and installed with pip install -e .
#                 --no-build-isolation. RoboTwin pins that tag rather than
#                 tracking the latest release. This builder pins that tag's full
#                 commit; --curobo-commit requires a full SHA. The original install
#                 reported version 0.0.0 because setuptools_scm could not read git
#                 metadata, so that string pins nothing.
#   warp-lang     the recovered lock keeps 1.10.1 while _install.sh asks for
#                 1.12.0. The recovered lock is the record of what actually ran
#                 and wins here.
#   setuptools    the recovered lock keeps 69.5.1, matching _install.sh, which
#                 overrides the 75.1.0 that conda installs.
#
# Asset archives are pinned to a Hugging Face commit and verified before
# extraction, including archives supplied through ROBOSCOPE_ROBOTWIN_ASSETS_DIR.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CUROBO_URL="https://github.com/NVlabs/curobo"
CUROBO_COMMIT="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSETS_SOURCE="TianxingChen/RoboTwin2.0"
ASSETS_REVISION="3dc3b798668feb99ac61cc9086d84cbcc3d79186"
INSTALL_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --curobo-commit) CUROBO_COMMIT="${2:?--curobo-commit requires a commit}"; shift 2 ;;
    --assets-source) ASSETS_SOURCE="${2:?--assets-source requires a dataset repo id}"; shift 2 ;;
    -*) echo "Unknown argument: $1" >&2; exit 2 ;;
    *)
      [[ -z "${INSTALL_ARG}" ]] || { echo "Only one <install-dir> may be given" >&2; exit 2; }
      INSTALL_ARG="$1"; shift ;;
  esac
done
[[ -n "${INSTALL_ARG}" ]] || {
  echo "usage: $0 [--curobo-commit <sha>] [--assets-source <dataset>] <install-dir>" >&2
  exit 2
}
install -d -m 0755 "${INSTALL_ARG}"
INSTALL_DIR="$(cd -- "${INSTALL_ARG}" && pwd -P)"

CONDA_SPECS="${SCRIPT_DIR}/conda-specs.txt"
REQUIREMENTS="${SCRIPT_DIR}/runtime-requirements.txt"
MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
LAUNCHER="${SCRIPT_DIR}/roboscope_robotwin_launcher.py"
SCANNER="${SCRIPT_DIR}/roboscope_robotwin_seed_scan.py"
SEED_ARCHIVE="${SCRIPT_DIR}/seed-cache-v2.tar.gz"
ENV_PREFIX="${INSTALL_DIR}/env"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
ROBOTWIN_ROOT="${INSTALL_DIR}/RoboTwin"
ROBOTWIN_URL="https://github.com/RoboTwin-Platform/RoboTwin"
KIT_PYTHON="${KIT_PYTHON:-python3}"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "RoboTwin runtime builds require Linux x86_64" >&2
  exit 3
fi
for required in "${CONDA_SPECS}" "${REQUIREMENTS}" "${MANIFEST}" "${LAUNCHER}" "${SCANNER}" "${SEED_ARCHIVE}"; do
  [[ -f "${required}" ]] || {
    echo "RoboTwin runtime input is missing: ${required}" >&2
    exit 3
  }
done
[[ "${CUROBO_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "--curobo-commit requires a 40-character commit SHA" >&2
  exit 3
}
grep -q '^python-3\.8\.' "${CONDA_SPECS}" || {
  echo "RoboTwin conda specs must pin CPython 3.8" >&2
  exit 3
}
for pin in "torch==2.4.1+cu121" "sapien==" "mplib==" "nvidia-curobo==0.0.0"; do
  grep -q "^${pin}" "${REQUIREMENTS}" || {
    echo "RoboTwin runtime lock must pin ${pin}" >&2
    exit 3
  }
done
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to build the RoboTwin runtime" >&2
  exit 3
fi
# The seed generations are built by the kit, so fail before the long build.
"${KIT_PYTHON}" -c 'import roboscope_kit' || {
  echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
  exit 3
}
# Ignore any pip configuration on the build host. GPU hosts commonly ship a
# /etc/pip.conf that adds an unreachable NVIDIA index, which pip then retries
# for every package, and a host-level index must not decide what a pinned
# runtime resolves to. The explicit index arguments below are the whole story.
export PIP_CONFIG_FILE=/dev/null
# cuRobo compiles CUDA kernels for several architectures at once. Parallel
# cicc processes exhaust memory on a normal build host and the kernel OOM
# killer takes them down, which surfaces only as "Error compiling objects".
# One job at a time builds reliably; override for a larger host.
MAX_JOBS="${ROBOSCOPE_BUILD_MAX_JOBS:-1}"
export MAX_JOBS
# Compile the same CUDA targets on every build host: Ada (8.9) is the GPU family
# the recorded results ran on, and Hopper (9.0) is also validated. Do not let the
# build host's detected GPU silently decide which GPUs the runtime can execute on.
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
# micromamba is a drop-in conda replacement, so accept either.
CONDA_BIN=""
for candidate in conda micromamba; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    CONDA_BIN="$(command -v "${candidate}")"
    break
  fi
done
[[ -n "${CONDA_BIN}" ]] || {
  echo "conda or micromamba is required to build the RoboTwin runtime" >&2
  exit 3
}
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# conda-specs.txt keeps the exact conda-meta directory names, which use the
# legacy name-version-build form. conda understands that spelling but micromamba
# parses the whole string as a package name, so normalise to name=version=build.
# The recovered build strings resolve only from the defaults channel, so that
# is the channel the original environment was built from.
readarray -t CONDA_PACKAGES < <(python3 - "${CONDA_SPECS}" <<'INNER'
import re
import sys

packages = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    match = re.match(r"^(.*)-([0-9][^-]*)-([^-]+)$", line)
    if not match:
        raise SystemExit(f"unparsable conda spec: {line}")
    name, version, build = match.groups()
    packages.append(f"{name}={version}={build}")
if not packages:
    raise SystemExit("no conda specs found")
print("\n".join(packages))
INNER
)
[[ "${#CONDA_PACKAGES[@]}" -gt 0 ]] || {
  echo "RoboTwin conda specs are empty" >&2
  exit 3
}

manifest_value() {
  python3 - "${MANIFEST}" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}
BASE_COMMIT="$(manifest_value benchmark_protocols.robotwin.robotwin_runtime_base_commit)"
RUNTIME_COMMIT="$(manifest_value benchmark_protocols.robotwin.robotwin_runtime_commit)"
SEED_ARCHIVE_SHA256="$(manifest_value runtime.seed_cache_archive_sha256)"
[[ "${BASE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "RoboTwin base commit is not a 40-character hash: ${BASE_COMMIT}" >&2
  exit 3
}

# Stage inside the install directory: the asset ZIPs alone are about 14 GB, and the
# finished tree then moves into place with a same-filesystem rename.
STAGING="$(mktemp -d "${INSTALL_DIR}/.robotwin-build.XXXXXX")"
trap 'rm -rf -- "${STAGING}"' EXIT

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  git clone --quiet --no-checkout "${ROBOTWIN_URL}" "${STAGING}/RoboTwin"
  git -C "${STAGING}/RoboTwin" fetch --quiet --depth 1 origin "${BASE_COMMIT}"
  git -C "${STAGING}/RoboTwin" checkout --quiet --detach "${BASE_COMMIT}"
  ACTUAL_BASE_COMMIT="$(git -C "${STAGING}/RoboTwin" rev-parse HEAD)"
  [[ "${ACTUAL_BASE_COMMIT}" == "${BASE_COMMIT}" ]] || {
    echo "RoboTwin checkout ${ACTUAL_BASE_COMMIT} differs from ${BASE_COMMIT}" >&2
    exit 5
  }

  # RoboTwin's sources use annotations that Python 3.8 evaluates at definition
  # time. Add the future import to every tracked module and commit it with fixed
  # identity and dates, so the result is exactly the protocol's runtime commit.
  python3 - "${STAGING}/RoboTwin" <<'PY'
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
tracked = subprocess.run(
    ("git", "-C", str(root), "ls-files", "-z", "*.py"),
    check=True,
    capture_output=True,
).stdout.split(b"\0")
paths = [root / value.decode() for value in tracked if value]
marker = b"from __future__ import annotations\n"
modified = 0
for path in paths:
    source = path.read_bytes()
    if marker in source:
        continue
    offset = source.find(b"\n") + 1 if source.startswith(b"#!") else 0
    path.write_bytes(source[:offset] + marker + source[offset:])
    modified += 1
if modified != 683:
    raise SystemExit(f"RoboTwin annotation patch changed {modified} files; expected 683")
PY
  git -C "${STAGING}/RoboTwin" add --update
  GIT_AUTHOR_NAME="RoboScope Runtime Builder" \
  GIT_AUTHOR_EMAIL="runtime@localhost" \
  GIT_AUTHOR_DATE="2026-08-29T00:00:00+08:00" \
  GIT_COMMITTER_NAME="RoboScope Runtime Builder" \
  GIT_COMMITTER_EMAIL="runtime@localhost" \
  GIT_COMMITTER_DATE="2026-08-29T00:00:00+08:00" \
    git -C "${STAGING}/RoboTwin" -c commit.gpgsign=false commit \
      --quiet -m "Apply RoboScope Python 3.8 annotation runtime patch"

  # The upstream assets/_download.py inside the pinned tree names this dataset and
  # restricts the download to these three archives with snapshot_download's
  # allow_patterns, so fetching exactly those three files is the same operation
  # without depending on huggingface_hub being importable on the build host. The
  # dataset has changed since the original tree was built; the pinned revision
  # defines the restored runtime rather than claiming the original bytes are known.
  ASSETS_URL_BASE="https://huggingface.co/datasets/${ASSETS_SOURCE}/resolve/${ASSETS_REVISION}"
  ASSETS_CACHE="${ROBOSCOPE_ROBOTWIN_ASSETS_DIR:-}"
  for archive in background_texture embodiments objects; do
    zip_path="${STAGING}/RoboTwin/assets/${archive}.zip"
    if [[ -n "${ASSETS_CACHE}" && -f "${ASSETS_CACHE}/${archive}.zip" ]]; then
      install -m 0644 "${ASSETS_CACHE}/${archive}.zip" "${zip_path}"
    else
      curl --fail --location --silent --show-error --output "${zip_path}" \
        "${ASSETS_URL_BASE}/${archive}.zip"
    fi
    case "${archive}" in
      background_texture) expected_sha256=54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070 ;;
      embodiments) expected_sha256=85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789 ;;
      objects) expected_sha256=6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27 ;;
    esac
    [[ "$(sha256sum "${zip_path}" | awk '{print $1}')" == "${expected_sha256}" ]] || {
      echo "RoboTwin ${archive} does not match pinned revision ${ASSETS_REVISION}" >&2
      exit 5
    }
    unzip -q -o "${zip_path}" -d "${STAGING}/RoboTwin/assets"
    # Match the pinned upstream download script: only extracted assets belong in
    # the runtime, rather than retaining another 14 GB of input ZIPs.
    rm -- "${zip_path}"
  done

  git clone --quiet --no-checkout "${CUROBO_URL}" "${STAGING}/RoboTwin/envs/curobo"
  git -C "${STAGING}/RoboTwin/envs/curobo" checkout --quiet --detach "${CUROBO_COMMIT}"
  # The editable cuRobo install below records the source location, so the tree
  # moves to its final path first.
  mv -- "${STAGING}/RoboTwin" "${ROBOTWIN_ROOT}"
fi

ACTUAL_RUNTIME_COMMIT="$(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_RUNTIME_COMMIT}" == "${RUNTIME_COMMIT}" ]] || {
  echo "RoboTwin tree is at ${ACTUAL_RUNTIME_COMMIT}, expected runtime commit ${RUNTIME_COMMIT}" >&2
  exit 5
}
for relative in assets/background_texture assets/embodiments assets/objects; do
  [[ -d "${ROBOTWIN_ROOT}/${relative}" ]] || {
    echo "RoboTwin asset directory is missing: ${relative}" >&2
    exit 5
  }
done
ACTUAL_CUROBO_COMMIT="$(git -C "${ROBOTWIN_ROOT}/envs/curobo" rev-parse HEAD)"
[[ "${ACTUAL_CUROBO_COMMIT}" == "${CUROBO_COMMIT}" ]] || {
  echo "cuRobo is at ${ACTUAL_CUROBO_COMMIT}, expected ${CUROBO_COMMIT}" >&2
  exit 5
}
[[ -f "${ROBOTWIN_ROOT}/envs/curobo/setup.py" || -f "${ROBOTWIN_ROOT}/envs/curobo/pyproject.toml" ]] || {
  echo "cuRobo source has no setup.py or pyproject.toml" >&2
  exit 5
}

if [[ ! -x "${ENV_PYTHON}" ]]; then
  "${CONDA_BIN}" create --yes --quiet -c defaults --prefix "${ENV_PREFIX}" "${CONDA_PACKAGES[@]}"
fi
# --no-deps: the recovered lock is the complete set of distributions the original
# environment had, so installing with dependency resolution can only substitute
# versions. torchvision 0.19.1 is paired with torch 2.4.1+cu121, which that wheel
# rejects; that state is what ran, so it is reproduced rather than resolved.
# nvidia-curobo is an editable install of the vendored cuRobo tree, and no index
# publishes the 0.0.0 the recovered lock records, so install the lock without it
# and let the editable install below provide it. torch 2.4.1+cu121 only exists on
# the PyTorch cu121 index.
grep -v '^nvidia-curobo==' "${REQUIREMENTS}" > "${STAGING}/requirements.txt"
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
  --extra-index-url "${TORCH_INDEX_URL}" \
  --requirement "${STAGING}/requirements.txt"

# RoboTwin's embodiment configs embed the absolute assets path; regenerate them
# from their templates for this install location.
(cd "${ROBOTWIN_ROOT}" && PYTHONNOUSERSITE=1 "${ENV_PYTHON}" script/update_embodiment_config_path.py)
PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - "${ROBOTWIN_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
templates = tuple((root / "assets/embodiments").rglob("*_tmp.yml"))
if not templates:
    raise SystemExit("RoboTwin embodiment path templates are missing")
for template in templates:
    expected = template.read_text().replace("${ASSETS_PATH}", str(root)).replace("$ASSETS_PATH", str(root))
    target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
    if not target.is_file() or target.read_text() != expected:
        raise SystemExit(f"RoboTwin embodiment path preparation failed: {target}")
print(f"RoboTwin embodiment paths verified: {len(templates)}")
PY

# cuRobo derives its version with setuptools_scm. A tagged checkout yields 0.7.8
# and a tree without git metadata makes setuptools_scm fail outright, so the
# 0.0.0 the recovered lock records is forced explicitly. cuRobo imports torch
# while generating its metadata, which is why the lock is installed first. The
# CUDA build takes a long time, so an existing 0.0.0 install is kept.
curobo_version() {
  PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I -c \
    'from importlib.metadata import version; print(version("nvidia-curobo"))' 2>/dev/null || true
}
if [[ "$(curobo_version)" != "0.0.0" ]]; then
  SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0" \
    PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -m pip install --no-cache-dir --no-deps \
    --no-build-isolation -e "${ROBOTWIN_ROOT}/envs/curobo"
fi
[[ "$(curobo_version)" == "0.0.0" ]] || {
  echo "cuRobo installed as $(curobo_version), expected 0.0.0" >&2
  exit 5
}

# Three site-packages files carry source patches; their sizes recorded in the
# 2026-09-02 inventory are the acceptance target:
#   sapien/wrapper/urdf_loader.py  the inventory records 29393 bytes and the
#                                   unpatched wheel is 29356. RoboTwin's own
#                                   script/_install.sh rewrites both open() calls
#                                   to pass encoding="utf-8", which is +36 bytes
#                                   and lands on the exact lines that script's
#                                   comments name, giving 29392. The remaining
#                                   byte is unexplained and is not invented here,
#                                   so the assertion below uses the value the
#                                   documented patch produces.
#   mplib/planner.py               32736 vs 32747 unpatched. The same script drops
#                                   "or collide " from the screw-plan rejection
#                                   test, which is -11 bytes and matches exactly.
#   mplib/urdf_utils.py            6165 vs 6130 unpatched. mplib 0.2.1 declares
#                                   Requires-Python >=3.8 but annotates with
#                                   str | Path, which Python 3.8 evaluates at
#                                   definition time and rejects. The future
#                                   import is +35 bytes and makes the module
#                                   importable at all.
#
# Each patch is a no-op on an already patched file, so re-runs are safe. A patch
# that stops matching fails the build instead of drifting silently.
SITE_PACKAGES="${ENV_PREFIX}/lib/python3.8/site-packages"
patch_runtime_file() {
  local relative="$1"
  local expected="$2"
  local path="${SITE_PACKAGES}/${relative}"
  [[ -f "${path}" ]] || {
    echo "RoboTwin runtime patch target is missing: ${path}" >&2
    exit 5
  }
  [[ "$(stat -c '%s' "${path}")" == "${expected}" ]] || {
    echo "RoboTwin runtime file ${relative} is $(stat -c '%s' "${path}") bytes, expected ${expected}" >&2
    exit 5
  }
}

sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
  "${SITE_PACKAGES}/sapien/wrapper/urdf_loader.py"
sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
  "${SITE_PACKAGES}/mplib/planner.py"
if ! grep -q '^from __future__ import annotations$' "${SITE_PACKAGES}/mplib/urdf_utils.py"; then
  sed -i '1i from __future__ import annotations' \
    "${SITE_PACKAGES}/mplib/urdf_utils.py"
fi
patch_runtime_file sapien/wrapper/urdf_loader.py 29392
patch_runtime_file mplib/planner.py 32736
patch_runtime_file mplib/urdf_utils.py 6165

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${ENV_PYTHON}" -I - "${ROBOTWIN_ROOT}" <<'PY'
import sys
from pathlib import Path

robotwin_root = Path(sys.argv[1])

import curobo
import mplib
import numpy
import sapien
import torch

from importlib.metadata import version

for distribution, expected in (
    ("torch", "2.4.1+cu121"),
    ("sapien", "3.0.0b1"),
    ("mplib", "0.2.1"),
    ("numpy", "1.24.4"),
):
    actual = version(distribution)
    if actual != expected:
        raise SystemExit(f"RoboTwin runtime {distribution} is {actual}, expected {expected}")

if not torch.cuda.is_available():
    raise SystemExit("RoboTwin runtime torch build has no CUDA support")

for relative in ("envs/curobo", "assets/background_texture", "assets/embodiments", "assets/objects"):
    if not (robotwin_root / relative).is_dir():
        raise SystemExit(f"RoboTwin tree is missing {relative}")

print("RoboTwin runtime imports passed")
PY

# Seed generations record the runtime commit, so tracked sources must be untouched.
[[ -z "$(git -C "${ROBOTWIN_ROOT}" status --short --untracked-files=no)" ]] || {
  echo "RoboTwin tracked sources differ from ${RUNTIME_COMMIT}" >&2
  exit 5
}

# The manifest names the launcher, scanner and seed archive by paths relative to
# itself, so they are copied next to it.
install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
install -m 0644 "${LAUNCHER}" "${INSTALL_DIR}/roboscope_robotwin_launcher.py"
install -m 0644 "${SCANNER}" "${INSTALL_DIR}/roboscope_robotwin_seed_scan.py"
install -m 0644 "${SEED_ARCHIVE}" "${INSTALL_DIR}/seed-cache-v2.tar.gz"
[[ "$(sha256sum "${INSTALL_DIR}/seed-cache-v2.tar.gz" | awk '{print $1}')" == "${SEED_ARCHIVE_SHA256}" ]] || {
  echo "RoboTwin frozen seed archive differs from the benchmark manifest" >&2
  exit 5
}
# The archive holds seed-cache/<suite>/<task>.pkl for both suites.
rm -rf -- "${INSTALL_DIR}/seed-cache"
tar -xzf "${INSTALL_DIR}/seed-cache-v2.tar.gz" --no-same-owner -C "${INSTALL_DIR}"

# Formal runs read seed-generations/<digest>/. Each generation's digest covers the
# protocol identity, the runtime commit, the suite's task config, the scanner's
# SHA-256 and every task's seeds, so matching the manifest's digest proves this
# install evaluates the frozen inputs.
for suite in demo_clean demo_randomized; do
  expected_digest="$(manifest_value "runtime.seed_generations.${suite}")"
  # Import main() rather than using -m: the robotwin package already imports
  # seed_protocol, which makes runpy warn.
  generation="$("${KIT_PYTHON}" -c \
    'from roboscope_kit.runtime.benchmarks.robotwin.seed_protocol import main; raise SystemExit(main())' \
    --source-cache-dir "${INSTALL_DIR}/seed-cache" \
    --generation-root "${INSTALL_DIR}/seed-generations" \
    --suite "${suite}" \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --scanner-path "${INSTALL_DIR}/roboscope_robotwin_seed_scan.py")"
  [[ "$(basename -- "${generation}")" == "${expected_digest}" ]] || {
    echo "RoboTwin ${suite} seed generation ${generation} differs from manifest digest ${expected_digest}" >&2
    exit 5
  }
  echo "RoboTwin ${suite} seed generation verified: ${expected_digest}"
done

# The self-test checks only what a quick-test needs; also confirm the installed
# manifest resolves the formal-run inputs to what this script produced.
"${KIT_PYTHON}" - "${INSTALL_DIR}/benchmark-manifest.json" <<'PY'
import sys

from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest

manifest = load_benchmark_manifest(sys.argv[1])
paths = [
    manifest.runtime_path(name)
    for name in ("seed_scanner_path", "seed_cache_dir", "seed_cache_archive_path")
]
root = manifest.runtime_path("seed_generation_root")
paths += [root / digest / "manifest.json" for digest in manifest.runtime["seed_generations"].values()]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"RoboTwin manifest names missing paths: {', '.join(missing)}")
PY

echo "Built RoboTwin runtime in ${INSTALL_DIR}"
echo "RoboTwin base commit ${BASE_COMMIT} -> runtime commit ${RUNTIME_COMMIT}"
echo "cuRobo at ${ACTUAL_CUROBO_COMMIT} installed at RoboTwin/envs/curobo"

"${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark robotwin --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
  --output-dir "${INSTALL_DIR}/self-test"
