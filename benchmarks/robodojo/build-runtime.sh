#!/usr/bin/env bash
set -Eeuo pipefail

# Build the pinned RoboDojo simulation runtime with its upstream installer.
# Usage: build-runtime.sh <install-dir>
BUILDER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${BUILDER_DIR}/../.." && pwd -P)"
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
  echo "RoboDojo runtime builds require Linux x86_64" >&2; exit 3;
}
case "${OMNI_KIT_ACCEPT_EULA:-}" in
  Y|YES|y|yes) ;;
  *)
    echo "RoboDojo uses NVIDIA Isaac Sim. Read its license and set OMNI_KIT_ACCEPT_EULA=YES to accept it:" >&2
    echo "https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html" >&2
    exit 3 ;;
esac
for executable in git curl sha256sum dpkg cmake ffmpeg; do
  command -v "${executable}" >/dev/null || { echo "Missing required executable: ${executable}" >&2; exit 3; }
done
for package in cmake build-essential ffmpeg; do
  dpkg -s "${package}" >/dev/null 2>&1 || {
    echo "The upstream installer requires the system package ${package}; install it before building." >&2
    exit 3
  }
done
mkdir -p "$1"
INSTALL_ROOT="$(cd -- "$1" && pwd -P)"
SOURCE_ROOT="${INSTALL_ROOT}/RoboDojo"
CONDA_ROOT="${INSTALL_ROOT}/miniforge"
ENV_ROOT="${CONDA_ROOT}/envs/RoboDojo"
SOURCE_URL="https://github.com/RoboDojo-Benchmark/RoboDojo.git"
SOURCE_COMMIT="f57e1633ffed59761bc8e568cfe7530c4b33381c"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh"
MINIFORGE_SHA256="65af53dad30b3fcbd1cb1d4ad62fd3a86221464754844544558aae3a28795189"

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --no-checkout "${SOURCE_URL}" "${SOURCE_ROOT}"
  git -C "${SOURCE_ROOT}" checkout --detach "${SOURCE_COMMIT}"
fi
[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]] || {
  echo "Existing RoboDojo checkout is not at ${SOURCE_COMMIT}: ${SOURCE_ROOT}" >&2; exit 5;
}

# The upstream script uses `submodule update --remote`. Give it local
# remote-tracking refs at the tested commits so that command remains pinned
# even when the upstream main branches advance.
git -C "${SOURCE_ROOT}" submodule update --init third_party/IsaacLab third_party/curobo XPolicyLab
pin_submodule() {
  local path="$1" commit="$2" branch="roboscope-pinned-$2"
  local checkout="${SOURCE_ROOT}/$1"
  if ! git -C "${checkout}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${checkout}" fetch origin "${commit}"
  fi
  git -C "${checkout}" config fetch.prune false
  git -C "${checkout}" config remote.origin.prune false
  git -C "${checkout}" update-ref "refs/remotes/origin/${branch}" "${commit}"
  git -C "${SOURCE_ROOT}" config "submodule.${path}.branch" "${branch}"
}
pin_submodule third_party/IsaacLab afca7b09d60d8beb9c1cb28b43066499940b969b
pin_submodule third_party/curobo 895c6517243f8cb091c73c018c8167192d39599a
pin_submodule XPolicyLab c7479d6fb014b0a23dfc292ef214eb1990f27029

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH PYTHONHOME VIRTUAL_ENV || true
installer="${INSTALL_ROOT}/miniforge-installer.sh"
if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  for ((attempt = 1; attempt <= 8; attempt++)); do
    if [[ -f "${installer}" ]] && printf '%s  %s\n' "${MINIFORGE_SHA256}" "${installer}" | sha256sum -c - >/dev/null 2>&1; then
      break
    fi
    curl -fLsS --retry 2 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
      -C - --output "${installer}" "${MINIFORGE_URL}" || true
  done
  printf '%s  %s\n' "${MINIFORGE_SHA256}" "${installer}" | sha256sum -c -
  bash "${installer}" -b -p "${CONDA_ROOT}"
fi
mkdir -p "${CONDA_ROOT}/envs" "${INSTALL_ROOT}/conda-pkgs" "${INSTALL_ROOT}/cache" "${INSTALL_ROOT}/tmp"
cat > "${INSTALL_ROOT}/condarc.yaml" <<EOF
channels:
  - conda-forge
default_channels: []
envs_dirs:
  - ${CONDA_ROOT}/envs
pkgs_dirs:
  - ${INSTALL_ROOT}/conda-pkgs
solver: classic
EOF
export PATH="${CONDA_ROOT}/bin:${PATH}"
export CONDARC="${INSTALL_ROOT}/condarc.yaml"
export CONDA_ENVS_PATH="${CONDA_ROOT}/envs" CONDA_PKGS_DIRS="${INSTALL_ROOT}/conda-pkgs"
# The upstream installer tries to accept Anaconda's terms when its plugin is
# present. This build uses conda-forge and does not accept those terms.
export CONDA_NO_PLUGINS=true CONDA_SOLVER=classic
export XDG_CACHE_HOME="${INSTALL_ROOT}/cache" PIP_CACHE_DIR="${INSTALL_ROOT}/cache/pip"
export TMPDIR="${INSTALL_ROOT}/tmp" PYTHONNOUSERSITE=1
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  conda create -p "${ENV_ROOT}" 'python=3.11.18' -y
fi
"${ENV_ROOT}/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 11, 18), "RoboDojo needs Python 3.11.18"'

# IsaacLab briefly installs starlette 0.49.1 before upstream pins Isaac Sim's
# 0.45.3. Constrain all other packages during installation and fix IsaacLab's
# metadata after the unchanged upstream script has completed.
awk 'tolower($0) !~ /^starlette==/' "${BUILDER_DIR}/runtime-requirements.txt" > "${INSTALL_ROOT}/bootstrap-constraints.txt"
export PIP_CONSTRAINT="${INSTALL_ROOT}/bootstrap-constraints.txt"
export ACCEPT_EULA=Y
# Upstream activates $HOME/miniconda3 before the Conda hook; use only the hook.
sed -i 's#source "$HOME/miniconda3/bin/activate" RoboDojo 2>/dev/null || ##' "${SOURCE_ROOT}/scripts/install.sh"
( cd "${SOURCE_ROOT}" && bash scripts/install.sh -i )
for spec in \
  'third_party/IsaacLab:afca7b09d60d8beb9c1cb28b43066499940b969b' \
  'third_party/curobo:895c6517243f8cb091c73c018c8167192d39599a' \
  'XPolicyLab:c7479d6fb014b0a23dfc292ef214eb1990f27029'; do
  path="${spec%%:*}" expected="${spec#*:}"
  [[ "$(git -C "${SOURCE_ROOT}/${path}" rev-parse HEAD)" == "${expected}" ]] || {
    echo "Unexpected RoboDojo submodule revision: ${path}" >&2; exit 5;
  }
done

apply_once() {
  local repository="$1" patch_file="$2" strip="${3:-1}"
  if git -C "${repository}" apply --check -p"${strip}" "${patch_file}"; then
    git -C "${repository}" apply -p"${strip}" "${patch_file}"
  elif ! git -C "${repository}" apply --reverse --check -p"${strip}" "${patch_file}"; then
    echo "Cannot apply ${patch_file}; the existing source differs from its expected state" >&2
    exit 5
  fi
}
apply_once "${SOURCE_ROOT}" "${BUILDER_DIR}/patches/layout-shards.patch"
apply_once "${SOURCE_ROOT}" "${BUILDER_DIR}/patches/policy-bridge.patch"
for name in async-video unpaced-render physics-threads control-state; do
  apply_once "${SOURCE_ROOT}" "${BUILDER_DIR}/patches/${name}.patch"
done
apply_once "${SOURCE_ROOT}/third_party/IsaacLab" "${BUILDER_DIR}/patches/isaaclab-deps.patch" 3
install -d "${SOURCE_ROOT}/XPolicyLab/policy/RoboScopeKit"
for file in __init__.py client.py deploy.py deploy.yml; do
  install -m 0644 "${BUILDER_DIR}/roboscope_policy/${file}" "${SOURCE_ROOT}/XPolicyLab/policy/RoboScopeKit/${file}"
done

python="${ENV_ROOT}/bin/python"
"${python}" -m pip install --no-deps --no-build-isolation -e "${SOURCE_ROOT}/third_party/IsaacLab/source/isaaclab"
"${python}" -m pip install pytest huggingface-hub 'pydantic>=2.5' 'h5py>=3.8' 'pyyaml>=6' 'msgpack>=1.0.8' msgpack-numpy
# A previous unpinned upstream installation may have left these behind after
# downgrading huggingface-hub; they conflict with Isaac Sim's pinned idna.
"${python}" -m pip uninstall -y httpx2 httpcore2
"${python}" -m pip install --no-deps "${REPO_DIR}/packages/roboscope-kit"
"${python}" -m pip check

# Verify every asset against the immutable upstream inventory. The optional
# mirror is a byte transport only; it cannot change the accepted hashes.
asset_args=(--transport "${ROBODOJO_ASSET_TRANSPORT:-primary}" --workers "${ROBODOJO_ASSET_WORKERS:-8}")
if [[ -n "${ROBODOJO_ASSET_INVENTORY:-}" ]]; then
  asset_args+=(--inventory "${ROBODOJO_ASSET_INVENTORY}")
fi
"${python}" "${BUILDER_DIR}/download_assets.py" "${SOURCE_ROOT}" "${asset_args[@]}"
( cd "${SOURCE_ROOT}" && "${python}" utils/update_embodiment_config_path.py )
if [[ -e "${INSTALL_ROOT}/env" && ! -L "${INSTALL_ROOT}/env" ]]; then
  echo "Existing env is not the RoboDojo environment symlink: ${INSTALL_ROOT}/env" >&2; exit 5
fi
ln -sfn "miniforge/envs/RoboDojo" "${INSTALL_ROOT}/env"
install -m 0644 "${BUILDER_DIR}/benchmark-manifest.json" "${INSTALL_ROOT}/benchmark-manifest.json"
"${python}" -m roboscope_kit.runtime.benchmark --benchmark robodojo --self-test \
  --benchmark-manifest "${INSTALL_ROOT}/benchmark-manifest.json" --output-dir "${INSTALL_ROOT}/self-test"
echo "RoboDojo runtime built at ${INSTALL_ROOT}"
