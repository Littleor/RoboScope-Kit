# LIBERO Evaluation — Autotune Handoff

## Purpose

`/root/startbenchmark.sh` is the sole external entrypoint. It now runs the
platform-selected Benchmark runtime whenever Cloud provides
`ROBOSCOPE_PLATFORM_ENTRYPOINT`. Without that variable it retains the hermetic
LIBERO self-test and legacy direct-run path.

The user-facing invocation, observation/action, episode, result, and error
contract is defined in
`packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md`. Keep this
handoff and the user manual consistent with that protocol whenever the runner
or shell entrypoint changes.

## Entrypoint

```bash
/root/startbenchmark.sh <suite> <checkpoint_source> [evaluation_num=50]
```

Suites: `libero_spatial` | `libero_object` | `libero_goal` | `libero_10` | `libero_90`

Flow:

1. Delegate to `ROBOSCOPE_PLATFORM_ENTRYPOINT` when Cloud selects a platform runtime
2. Otherwise run `resetenv.sh --soft` before opening the current run log
3. Download an OSS checkpoint to `/root/benchmark/checkpoint`, or use an absolute image-local checkpoint path
4. In `policy` runtime mode, load and validate `/workspace/roboscope-policy.json`, start the declared user Policy entrypoint, and pass the staged checkpoint through `SessionContext`
5. Otherwise resolve StarVLA metadata and start the existing StarVLA server path
6. Run persistent LIBERO workers through the selected inference protocol
7. Publish aggregate progress through `/root/testlog.txt` until `finish` or `error:*`

## Runtime note

`roboscope_kit.runtime` owns benchmark orchestration and K/g selection.
SDK 0.7.0b58 and newer require Python 3.11 or newer in every environment
that imports the SDK. The previously pinned ``libero`` Python 3.10 environment
is unsupported and must be upgraded and accepted separately before using
this release. The Control Python 3.11 runtime alone does not upgrade simulator
or Policy interpreters.

## Image layout

| Path | Role |
| --- | --- |
| `/root/startbenchmark.sh` | External CLI |
| `/root/resetenv.sh` | Pre-run process/workdir cleanup |
| `/root/roboscope_libero_runner.py` | OSS download + asset staging + orchestration |
| `/root/roboscope_libero_autotune_eval.py` | Autotune bridge + legacy log mirroring |
| `/root/roboscope-kit/packages/roboscope-kit/` | Public Policy API + platform runtime |
| `/root/roboscope-kit/benchmarks/libero_official/starvla_runtime/` | StarVLA LIBERO runtime |
| `/opt/roboscope/control-runtime/` | Image-local Control Python + complete platform runtime |
| `/root/roboscope_assets/base_assets.json` | Base-model OSS map |
| `/root/starVLA` | Policy server + `ModelClient` |
| `/root/LIBERO` | Official sim |
| `/root/miniconda3/envs/{starVLA,libero}` | Interpreter split |

Source of truth in git:

- `roboscope-kit/benchmarks/libero_official/`
- `roboscope-kit/packages/roboscope-kit/`

Do not sync those paths by hand.

## Product status protocol

### `/root/testlog.txt`

```
用户模型权重下载进度 <0-100>%
测试套件下载进度 <0-100>%
LIBERO <suite> <total> <success> <failed> <status>
```

`status` values: `idle` | `downloading` | `running` | `finish` | `error:<code>`

Heartbeat rewrites the file every 5s during a run.

### Other artifacts

- Videos: `/root/testvideo/`
- Autotune internals: `/root/testresult/autotune/` (scored result and worker/server logs)
- Probe internals: `/root/testresult/autotune/probe/` (`metrics.json`, `events.jsonl`, client/server logs)

## Autotune defaults / knobs

The default path probes the current checkpoint and allocated CPU/GPU resources,
then selects stable worker concurrency (`K`) and policy replicas (`g`) before
starting the scored episodes:

| Env | Default | Meaning |
| --- | --- | --- |
| `ROBOSCOPE_KIT_ROOT` | `/root/roboscope-kit` | Autotune package root |
| `BENCHMARK_GPU_COUNT` | auto via `nvidia-smi` | GPU count |
| `BENCHMARK_REPLICAS` | auto-probed from measured VRAM and throughput | Legacy StarVLA-only fixed replica candidate |
| `BENCHMARK_CONCURRENCY` | auto-probed | Legacy StarVLA-only fixed worker candidate |
| `BENCHMARK_AUTOTUNE_PROBE_DURATION` | `10` | Single-client baseline window (s) |
| `BENCHMARK_AUTOTUNE_TRIAL_DURATION` | `20` | Candidate measurement window (s) |
| `BENCHMARK_AUTOTUNE_EPISODES_PER_WORKER` | `2` | Full episodes per SDK Policy worker and candidate |
| `BENCHMARK_CHUNK_SIZE` | `1` | Episodes per queue claim |
| `BENCHMARK_SERVER_CPU_THREADS` | `2` | Threads per policy server |
| `BENCHMARK_CLIENT_CPU_THREADS` | `1` | Threads per LIBERO worker |
| `BENCHMARK_ALLOW_EPISODE_SHARDING` | `1` | Allow `K > num_tasks` |
| `BENCHMARK_POLICY_PORT` | `16694` | Base websocket port |
| `BENCHMARK_SEED` | `7` | Env seed |
| `BENCHMARK_RENDER_BACKEND` | `egl` | `egl` or `osmesa` |
| `BENCHMARK_READY_TIMEOUT` | `900` | Server/worker ready and SDK `Policy.load()` timeout (s) |

SDK-native user policy mode uses the fixed image contract:

```json
{
  "schema_version": 2,
  "command": ["/opt/user/bin/python", "/workspace/user_policy.py"],
  "cwd": "/workspace"
}
```

The file path is always `/workspace/roboscope-policy.json`; it is not user-configurable. Platform-owned runtime variables are:

| Env | Meaning |
| --- | --- |
| `ROBOSCOPE_BENCHMARK_RUNTIME` | Exact runtime selector: `policy` for the SDK-native path; the StarVLA task leaves it unset |

The SDK-native scored path measures real Policy-backed LIBERO episodes across
paired K/g candidates, with one active evaluator worker per isolated Policy
replica. It stops when episode throughput no longer improves or CPU/GPU resource
signals approach their limits. Candidate measurements, rejection reasons, and
the final selection are written under `/root/testresult/autotune/probe/`; users
do not declare a concurrency capacity in `PolicyCapabilities`.

`--quick-test` fixes `K=1/g=1` and skips parameter probing so the user-level
smoke check stays cheap. Scored evaluation uses dynamic probing by default.

OSS credential envs are unchanged from legacy:

- Checkpoint: `BENCHMARK_CKPT_ENDPOINT`, `BENCHMARK_CKPT_ACCESS_KEY_ID`, `BENCHMARK_CKPT_ACCESS_KEY_SECRET`, optional `BENCHMARK_CKPT_SECURITY_TOKEN`
- StarVLA metadata: `BENCHMARK_BASE_*` plus `BENCHMARK_BASE_OSS_URI` (for example `oss://<bucket>/benchmark/basemodel`); weight files are excluded

For a custom image, bake the complete checkpoint directory into
`/root/roboscope-checkpoint` and pass that absolute path to the entrypoint. This
skips only the user-checkpoint OSS transfer; StarVLA metadata assets still use
OSS and continue to exclude model weights.

## Protocol note (scoring vs throughput)

With `BENCHMARK_ALLOW_EPISODE_SHARDING=1` (default), multiple workers may run
episodes from the same task. That maximizes throughput but changes simulator RNG
progression versus the single-process StarVLA evaluator, so success counts are
**high-throughput evaluation results**, not a bit-exact official leaderboard
reproduction.

For closer-to-legacy scoring:

```bash
export BENCHMARK_ALLOW_EPISODE_SHARDING=0
export BENCHMARK_CONCURRENCY=10   # <= task count for the suite
```

## StarVLA construction metadata

`config.yaml` `framework.qwenvl.base_vlm` basename must exist in
`base_assets.json`.

The referenced OSS directory supplies only Hugging Face config, tokenizer, and
processor files. The runner excludes `.safetensors`, `.bin`, `.pt`, `.pth`,
`.ckpt`, and weight-index files. All model parameters come from the uploaded
user checkpoint through a strict state-dict assignment. The small `fast`
processor asset is prepared only when `action_model_type` actually uses FAST.

Tokenizer distinctions still matter:

- `Qwen2.5-VL-3B-Instruct` — plain HF Instruct (`embed` 151936). Required by
  QwenOFT checkpoints such as `Qwen2.5-VL-OFT-LIBERO-4in1`.
- `Qwen2.5-VL-3B-Instruct-Action` — Instruct + 2048 `<robot_action_*>` tokens
  (`embed` 153713). Required by FAST / discrete-action paths.

Do **not** alias Instruct → Action. They are not interchangeable under
`load_state_dict(strict=True)`.

Ensure OSS contains the metadata files under:

`oss://<bucket>/benchmark/basemodel/Qwen2.5-VL-3B-Instruct/`

before running OFT configs that point at
`./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct`.

## Smoke checks

```bash
# Hermetic SDK/entrypoint check with no OSS, GPU, dataset or checkpoint.
benchmarks/libero_official/startbenchmark.sh --self-test

# User-level quick test. Cloud must inject BENCHMARK_CKPT_* and
# BENCHMARK_BASE_*; this validates OSS access, loads the real model and runs
# task 0 for one episode with one server/worker and no video.
/root/startbenchmark.sh --quick-test libero_10 oss://<bucket>/benchmark/usr/.../checkpoint/

# Imports (no GPU required)
/root/miniconda3/envs/libero/bin/python - <<'PY'
import sys
sys.path.insert(0, "/root")
sys.path.insert(0, "/root/roboscope-kit")
from roboscope_libero_autotune_eval import build_autotune_args
from benchmarks.libero_official.starvla_runtime import libero_server
print("StarVLA LIBERO runtime imports ok")
PY

# Entrypoint usage / missing-args log shape
/root/startbenchmark.sh; sed -n '1,4p' /root/testlog.txt

# Full run (needs GPUs + OSS creds + Instruct base asset for OFT)
/root/startbenchmark.sh libero_10 oss://<bucket>/benchmark/usr/.../checkpoint/ 50
/root/startbenchmark.sh libero_10 /root/roboscope-checkpoint 50
tail -f /root/testlog.txt
```

## Error codes surfaced in `testlog.txt`

Same style as legacy (`error:<code>`), including:

`missing_args` · `missing_autotune_runtime` · `missing_oss_env` ·
`ckpt_download_failed` · `base_download_failed` · `invalid_checkpoint_layout` ·
`invalid_checkpoint_source` · `local_checkpoint_missing` ·
`missing_base_vlm` · `unknown_base_model` · `evaluation_failed` ·
`invalid_suite` · `invalid_evaluation_num`
