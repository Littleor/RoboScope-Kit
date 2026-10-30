#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_ROOT="/root"
DEFAULT_PYTHON=""
STARVLA_PYTHON=""
SKIP_SELF_TEST=0

usage() {
  cat <<'EOF'
Usage: verify_official_image.sh [--target-root <absolute-path>]
                                [--default-python <path>]
                                [--starvla-python <path>]
                                [--skip-self-test]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root)
      TARGET_ROOT="${2:?--target-root requires an absolute path}"
      shift 2
      ;;
    --default-python)
      DEFAULT_PYTHON="${2:?--default-python requires a path}"
      shift 2
      ;;
    --starvla-python)
      STARVLA_PYTHON="${2:?--starvla-python requires a path}"
      shift 2
      ;;
    --skip-self-test)
      SKIP_SELF_TEST=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${TARGET_ROOT}" != /* || "${TARGET_ROOT}" == "/" ]]; then
  echo "Target root must be an absolute directory other than /" >&2
  exit 3
fi
if [[ -z "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="${TARGET_ROOT}/miniconda3/bin/python"
fi
if [[ -z "${STARVLA_PYTHON}" ]]; then
  STARVLA_PYTHON="${TARGET_ROOT}/miniconda3/envs/starVLA/bin/python"
fi

TARGET_METADATA_ROOT="${TARGET_ROOT}/.roboscope-official-image"
IMAGE_ROOT="$(cd -- "${TARGET_ROOT}/.." && pwd)"
CONTROL_RUNTIME_ROOT="${IMAGE_ROOT%/}/opt/roboscope/control-runtime"
MANIFEST="${TARGET_METADATA_ROOT}/manifest.json"
RUNTIME_CHECKSUMS="${TARGET_METADATA_ROOT}/runtime.sha256"

for path in \
  "${MANIFEST}" \
  "${RUNTIME_CHECKSUMS}" \
  "${TARGET_METADATA_ROOT}/verify-control-runtime.sh" \
  "${TARGET_ROOT}/startbenchmark.sh" \
  "${TARGET_ROOT}/resetenv.sh" \
  "${TARGET_ROOT}/../workspace/roboscope-policy.json" \
  "${TARGET_ROOT}/../opt/roboscope/defaults/roboscope-policy.json" \
  "${TARGET_ROOT}/../opt/roboscope/starvla/starvla_official_policy.py" \
  "${TARGET_ROOT}/../opt/roboscope/starvla/starvla_base_assets.py" \
  "${TARGET_ROOT}/../opt/roboscope/starvla/starvla_full_checkpoint.py" \
  "${TARGET_ROOT}/../opt/roboscope/starvla/export_starvla_eval_run.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing official image path: ${path}" >&2
    exit 4
  fi
done

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${TARGET_ROOT}" && sha256sum -c "${RUNTIME_CHECKSUMS}" >/dev/null)
else
  (cd "${TARGET_ROOT}" && shasum -a 256 -c "${RUNTIME_CHECKSUMS}" >/dev/null)
fi
echo "Installed runtime checksums passed"

TRUE_COMMAND="$(type -P true)"
ROBOSCOPE_PLATFORM_ENTRYPOINT="${TRUE_COMMAND}" "${TARGET_ROOT}/startbenchmark.sh"
ROBOSCOPE_PLATFORM_RESET="${TRUE_COMMAND}" "${TARGET_ROOT}/resetenv.sh"
echo "Platform Benchmark delegation verified"

if [[ -e "${TARGET_ROOT}/roboscope-kit" ]]; then
  echo "StarVLA image contains a stale RoboScope Kit checkout: ${TARGET_ROOT}/roboscope-kit" >&2
  exit 4
fi

CONTROL_METADATA="$("${DEFAULT_PYTHON}" - "${MANIFEST}" "${CONTROL_RUNTIME_ROOT}" <<'PY'
import json
import sys

manifest_path, control_runtime_root = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
control_runtime = manifest["control_runtime"]
if control_runtime.get("root") != control_runtime_root:
    raise SystemExit("StarVLA image manifest has an invalid Control Runtime root")
if control_runtime.get("python") != f"{control_runtime_root}/control-python/bin/python":
    raise SystemExit("StarVLA image manifest has an invalid Control Runtime python")
artifacts = manifest.get("control_runtime_artifacts", {})
if control_runtime.get("python_version") != artifacts.get("python", {}).get("version"):
    raise SystemExit("StarVLA image manifest Control Runtime python version mismatch")
if manifest["python_environments"]["control"] != manifest["control_runtime"]["python"]:
    raise SystemExit("StarVLA image manifest has inconsistent Control Python metadata")
print(manifest["evaluation_kit_commit"])
print(manifest["sdk"]["version"])
PY
)"
mapfile -t CONTROL_METADATA_LINES <<< "${CONTROL_METADATA}"
if [[ "${#CONTROL_METADATA_LINES[@]}" -ne 2 ]]; then
  echo "StarVLA image manifest has incomplete Control Runtime metadata" >&2
  exit 4
fi
KIT_COMMIT="${CONTROL_METADATA_LINES[0]}"
SDK_VERSION="${CONTROL_METADATA_LINES[1]}"

"${TARGET_METADATA_ROOT}/verify-control-runtime.sh" \
  --verify-only \
  --target-root "${CONTROL_RUNTIME_ROOT}" \
  --kit-commit "${KIT_COMMIT}" \
  --sdk-version "${SDK_VERSION}"

verify_sdk() {
  local python_path="$1"
  "${python_path}" - "${SDK_VERSION}" <<'PY'
import importlib.metadata
import importlib.util
import sys

import roboscope_kit
from roboscope_kit.policy import SessionContext

expected = sys.argv[1]
actual = importlib.metadata.version("roboscope-kit")
if actual != expected:
    raise SystemExit(f"SDK version mismatch: expected {expected}, got {actual}")
for module_name in (
    "roboscope_kit.runtime",
    "roboscope_kit.benchmark_specs",
    "roboscope_kit.client",
    "roboscope_kit.policy_config",
):
    if importlib.util.find_spec(module_name) is not None:
        raise SystemExit(f"Policy-core SDK unexpectedly contains {module_name}")
if hasattr(SessionContext, "create"):
    raise SystemExit("Policy-core SDK unexpectedly exposes registry-dependent SessionContext.create")
print(f"{sys.executable}: roboscope-kit {actual} (policy core)")
PY
}

verify_sdk "${STARVLA_PYTHON}"

PYTHONPATH="${TARGET_ROOT}/../opt/roboscope/starvla" \
  "${STARVLA_PYTHON}" - <<'PY'
from starvla_base_assets import ensure_starvla_base_model
from starvla_full_checkpoint import load_full_checkpoint
from starvla_official_policy import StarVLAOfficialPolicy, resolve_starvla_run_snapshot

assert ensure_starvla_base_model and StarVLAOfficialPolicy and load_full_checkpoint and resolve_starvla_run_snapshot
from starvla_official_policy import _image_preprocessor

assert _image_preprocessor("metaworld") is not None
print("Benchmark-independent StarVLA Policy imports passed")
PY

"${DEFAULT_PYTHON}" - "${MANIFEST}" "${TARGET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
target_root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config_path = target_root.parent / "workspace" / "roboscope-policy.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
if config["command"][-1] != manifest["starvla"]["policy"]:
    raise SystemExit(
        f"Policy config command mismatch: {config['command'][-1]} != {manifest['starvla']['policy']}"
    )
if config.get("cwd") != manifest["starvla"]["cwd"]:
    raise SystemExit(f"Policy working directory must be {manifest['starvla']['cwd']}")
print("StarVLA Policy configuration verified")
PY

if [[ "${SKIP_SELF_TEST}" != "1" ]]; then
  if [[ "${TARGET_ROOT}" != "/root" ]]; then
    echo "The image self-test must run against the production /root layout" >&2
    exit 5
  fi
  for path in \
    "${TARGET_ROOT}/starVLA" \
    "${TARGET_ROOT}/miniconda3/envs/starVLA"; do
    if [[ ! -d "${path}" ]]; then
      echo "Missing pre-baked StarVLA dependency: ${path}" >&2
      exit 6
    fi
  done
  (
    cd "${TARGET_ROOT}/../opt/roboscope/starvla"
    PYTHONPATH="${TARGET_ROOT}/../opt/roboscope/starvla" \
      "${STARVLA_PYTHON}" - <<'PY'
import sys
from pathlib import Path

import starvla_official_policy
from starvla_full_checkpoint import validate_starvla_import_root

expected = Path("/root/starVLA")
if Path(sys.path[0]) != expected:
    raise SystemExit(f"StarVLA Policy did not pin its import root: {sys.path[0]}")
validate_starvla_import_root()
print(f"StarVLA Policy import root {expected} verified")
PY
  )
  (
    cd "${TARGET_ROOT}/starVLA" &&
    PYTHONPATH="${TARGET_ROOT}/../opt/roboscope/starvla" \
      "${STARVLA_PYTHON}" - <<'PY'
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)

if "metaworld_mt50" not in DATASET_NAMED_MIXTURES:
    raise SystemExit("MetaWorld data registry is missing from the pinned StarVLA runtime")
if "metaworld_robot" not in ROBOT_TYPE_CONFIG_MAP:
    raise SystemExit("MetaWorld robot DataConfig is not registered")
print("MetaWorld data registry verified in the pinned StarVLA runtime")
PY
  )
  "${DEFAULT_PYTHON}" - "${MANIFEST}" "${TARGET_ROOT}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
target_root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_commit = manifest["starvla"]["commit"]
actual_commit = subprocess.check_output(
    ["git", "-C", str(target_root / "starVLA"), "rev-parse", "HEAD"],
    text=True,
).strip()
if actual_commit != expected_commit:
    raise SystemExit(f"StarVLA commit mismatch: expected {expected_commit}, got {actual_commit}")
print(f"StarVLA {actual_commit} verified")
PY
  # Run from the StarVLA root (matching the Policy's cwd) so ``import starVLA``
  # resolves the pinned source, with the official modules on PYTHONPATH.
  (
    cd "${TARGET_ROOT}/starVLA" &&
    PYTHONPATH="${TARGET_ROOT}/../opt/roboscope/starvla" \
      "${STARVLA_PYTHON}" - <<'PY'
from starvla_full_checkpoint import (
    _register_stable_mixture_aliases,
    _stable_mixture_aliases,
)

# Apply the aliases against the real pinned runtime, then assert they landed in
# the dict the dataset-mixture resolver actually reads: registry keeps its own
# copy of the base mixtures, so aliasing the base module would not reach it.
_register_stable_mixture_aliases()
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

# Recomputing over the resolver's own registry must now find nothing left to
# alias. This mirrors the production helper exactly (including its ambiguity and
# already-registered guards), so a legitimately ambiguous key never fails the
# build, while a fix that updated the wrong dict leaves work outstanding here.
remaining = _stable_mixture_aliases(DATASET_NAMED_MIXTURES)
if remaining:
    raise SystemExit(f"stable mixture ABI aliases missing in resolver registry: {sorted(remaining)}")
print("Stable dataset-mixture ABI aliases verified in resolver registry")
PY
  )
fi

"${DEFAULT_PYTHON}" - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
print(
    "StarVLA official image verified: "
    f"evaluation_kit={manifest['evaluation_kit_commit']} sdk={manifest['sdk']['version']}"
)
PY
