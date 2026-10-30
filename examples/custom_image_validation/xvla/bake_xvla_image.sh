#!/usr/bin/env bash
# Bake the X-VLA custom-image validation demo.
#
# Run on a fresh AutoDL instance after installing the commit-pinned official
# StarVLA bundle. The X-VLA image must share the Control Runtime generation of
# the live shared-FS thin launchers or its evaluations abort on the manifest
# check; the bundle's ``KIT_COMMIT`` is verified against
# /opt/roboscope/control-runtime/KIT_COMMIT below as an operational gate (it
# does not replace the clean-checkout guarantee of bundle provenance).
# Installs a dedicated ``xvla`` conda env
# (python 3.10 + torch 2.1 cu121 + transformers <= 4.51.3), the X-VLA repo and
# checkpoints, the policy adapter, and /workspace/roboscope-policy.json.  Also
# applies the X-VLA-only runtime settings (LIBERO absolute control, CALVIN
# absolute tuple, RoboTwin EE control). RoboTwin is selected by the image
# entrypoint before the platform runtime starts; the generic SDK source remains
# unmodified. After the script finishes, power the instance off and save it as
# a private image.
#
# Usage: bash bake_xvla_image.sh
set -euo pipefail

XVLA_ROOT=/opt/roboscope/xvla
XREPO="$XVLA_ROOT/X-VLA"
CKPT_ROOT="$XVLA_ROOT/checkpoints"
BUNDLE_DIR=/root/bake-xvla-bundle   # scp target for this directory
POLICY_SRC="$BUNDLE_DIR/xvla_demo_policy.py"
ENTRYPOINT_SRC="$BUNDLE_DIR/startbenchmark.sh"
KIT_COMMIT_FILE="$BUNDLE_DIR/KIT_COMMIT"
CONTROL_KIT_COMMIT_FILE=/opt/roboscope/control-runtime/KIT_COMMIT

if [ ! -f "$KIT_COMMIT_FILE" ]; then
  echo "missing bundle RoboScope Kit commit: $KIT_COMMIT_FILE"
  exit 1
fi
if [ ! -f "$POLICY_SRC" ]; then
  echo "missing bundled X-VLA policy: $POLICY_SRC"
  exit 1
fi
if [ ! -f "$ENTRYPOINT_SRC" ]; then
  echo "missing bundled X-VLA entrypoint: $ENTRYPOINT_SRC"
  exit 1
fi
EXPECTED_KIT_COMMIT=$(<"$KIT_COMMIT_FILE")
if [[ ! "$EXPECTED_KIT_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid bundle RoboScope Kit commit: $EXPECTED_KIT_COMMIT"
  exit 1
fi
if [ ! -f "$CONTROL_KIT_COMMIT_FILE" ]; then
  echo "missing image Control Runtime commit: $CONTROL_KIT_COMMIT_FILE"
  exit 1
fi
CONTROL_KIT_COMMIT=$(<"$CONTROL_KIT_COMMIT_FILE")
if [ "$CONTROL_KIT_COMMIT" != "$EXPECTED_KIT_COMMIT" ]; then
  echo "control runtime commit mismatch: $CONTROL_KIT_COMMIT != $EXPECTED_KIT_COMMIT"
  exit 1
fi

# expected sha256 of each checkpoint model.safetensors
declare -A CKPT_SHA=(
  [calvin]=98813bb2063fa62f8ffb21696546005b61defca6c0724adeab524c8f82f84bd4
  [libero]=260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8
  [robotwin]=3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b
)

echo "== [1/8] miniconda (detect or install) =="
if [ ! -x /root/miniconda3/bin/conda ]; then
  echo "installing Miniconda3 (py310) ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /root/miniconda3
fi
export PATH=/root/miniconda3/bin:$PATH
conda config --set channel_priority strict

echo "== [2/8] xvla conda env =="
if [ ! -d /root/miniconda3/envs/xvla ]; then
  # tsinghua pkgs/main only; conda-forge mirror is flaky from this network
  for i in 1 2 3 4; do
    if timeout 300 conda create -y -n xvla python=3.10 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main; then
      break
    fi
    echo "conda create attempt $i failed, retrying"
    sleep 10
  done
fi
source /root/miniconda3/bin/activate xvla
python -V

echo "== [3/8] python deps (torch cu121 via PyPI mirror) =="
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2
pip install \
  "transformers==4.51.3" \
  "peft==0.17.1" \
  "accelerate==1.2.1" \
  "numpy==1.26.3" \
  "scipy==1.15.0" \
  "safetensors==0.4.5" \
  "einops==0.8.1" \
  "timm==1.0.12" \
  "mmengine==0.10.5" \
  "json_numpy==2.1.0" \
  "fastapi" "uvicorn==0.34.3" \
  "opencv-python-headless" \
  "requests" "av==15.0.0" "h5py==3.12.1" "pyarrow==20.0.0" \
  "sapien==3.0.0b1" "mplib==0.2.1" "mujoco" \
  "setuptools<81"

echo "== [4/8] X-VLA repo =="
mkdir -p "$XVLA_ROOT"
if [ ! -d "$XREPO/.git" ]; then
  git config --global --unset-all "url.https://gh.llkk.cc/https://github.com/.insteadOf" 2>/dev/null || true
  git clone --depth 1 https://github.com/2toINF/X-VLA.git "$XREPO"
fi
python -c "import sys; sys.path.insert(0, '$XREPO'); import models.modeling_xvla; print('XVLA import OK')"

echo "== [5/8] checkpoints (single connection + sha verify; aria2 multi-segment corrupts) =="
mkdir -p "$CKPT_ROOT"
for ck in calvin libero robotwin; do
  repo=$( [ "$ck" = "calvin" ] && echo "2toINF/X-VLA-Calvin-ABC_D" || ( [ "$ck" = "libero" ] && echo "2toINF/X-VLA-Libero" || echo "2toINF/X-VLA-RoboTwin2" ))
  mkdir -p "$CKPT_ROOT/$ck"
  cd "$CKPT_ROOT/$ck"
  echo "-- $ck model.safetensors"
  if [ -f model.safetensors ] && [ "$(sha256sum model.safetensors | cut -c1-64)" = "${CKPT_SHA[$ck]}" ]; then
    echo "-- $ck model.safetensors already present with matching sha256, skipping download"
  else
    rm -f model.safetensors
    curl --http1.1 -L -C - --retry 20 --retry-delay 5 --retry-all-errors --max-time 6000 \
      -o model.safetensors "https://hf-mirror.com/$repo/resolve/main/model.safetensors"
    got=$(sha256sum model.safetensors | cut -c1-64)
    if [ "$got" != "${CKPT_SHA[$ck]}" ]; then
      echo "SHA MISMATCH for $ck: $got != ${CKPT_SHA[$ck]}"
      exit 1
    fi
  fi
  echo "-- $ck small files"
  for f in config.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json \
           configuration_xvla.py configuration_florence2.py modeling_xvla.py modeling_florence2.py \
           processing_xvla.py action_hub.py transformer.py; do
    curl -sL --retry 5 --retry-delay 2 --max-time 60 -o "$f" "https://hf-mirror.com/$repo/resolve/main/$f"
  done
done

echo "== [6/8] SDK install + faithful-recipe runtime settings =="
# the control runtime python needs roboscope_kit for the policy import
pip install -q /opt/roboscope/control-runtime/packages/roboscope-kit
python3 - <<'PY'
import re
from pathlib import Path

sdk = Path("/opt/roboscope/control-runtime/packages/roboscope-kit")
for root in (sdk / "src", sdk / "build" / "lib"):
    if not root.exists():
        continue
    # LIBERO: absolute EE control (X-VLA recipe) after settle, recipe horizons.
    wp = root / "roboscope_kit/runtime/benchmarks/libero/worker.py"
    if wp.exists():
        src = wp.read_text()
        src = re.sub(
            r"MAX_STEPS = \{[^}]*\}",
            'MAX_STEPS = {\n    "libero_spatial": 800,\n    "libero_object": 800,\n    "libero_goal": 800,\n    "libero_10": 900,\n    "libero_90": 800,\n}',
            src,
            count=1,
        )
        anchor = "        for _ in range(settle_steps):\n            observation, _reward, task_done, _info = runtime.env.step(LIBERO_DUMMY_ACTION)\n            if task_done or runtime.env.env.done:\n                return bool(task_done), 0, time.perf_counter() - started, frames"
        patch = anchor + "\n        # Faithful to the X-VLA libero_client.py recipe: absolute EE control.\n        runtime.env.env.robots[0].controller.use_delta = False\n"
        assert anchor in src
        src = src.replace(anchor, patch, 1)
        wp.write_text(src)
        print("patched", wp)
PY
# clear stale pyc
find /opt/roboscope/control-runtime -path "*benchmarks/libero/__pycache__*" -name "*.pyc" -delete 2>/dev/null || true
find /opt/roboscope/control-runtime -path "*benchmarks/calvin/__pycache__*" -name "*.pyc" -delete 2>/dev/null || true
find /opt/roboscope/control-runtime -path "*benchmarks/robotwin/__pycache__*" -name "*.pyc" -delete 2>/dev/null || true

echo "== [7/8] policy adapter + config =="
cp "$POLICY_SRC" "$XVLA_ROOT/xvla_demo_policy.py"
install -m 0755 "$ENTRYPOINT_SRC" /root/startbenchmark.sh
mkdir -p /workspace
cat > /workspace/roboscope-policy.json <<'JSON'
{
  "schema_version": 2,
  "command": [
    "/root/miniconda3/envs/xvla/bin/python",
    "/opt/roboscope/xvla/xvla_demo_policy.py"
  ],
  "cwd": "/opt/roboscope/xvla/X-VLA"
}
JSON

echo "== [8/8] cleanup transient state =="
rm -rf /root/testresult /root/testlog.txt /root/benchmark /root/.cache/huggingface
rm -rf /root/bake-xvla-bundle
echo "bake complete. Power the instance off and save the image."
