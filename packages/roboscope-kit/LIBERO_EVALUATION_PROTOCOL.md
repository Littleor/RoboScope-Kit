# LIBERO Evaluation Protocol

This document defines the external RoboScope LIBERO evaluation contract for custom inference code. It is
the protocol implemented by `/root/startbenchmark.sh`, the LIBERO runner, and the SDK-native LIBERO
adapter/worker.

The current protocol identifiers are:

- benchmark specification: `libero-v1`;
- SDK-native evaluator transport: `roboscope-policy-v1`;
- legacy product progress files: `legacy-startbenchmark-v1`.

SDK 0.7.0b58 and newer require Python 3.11 or newer in every process that
imports the SDK, including Policy and simulator workers. Python 3.8–3.10
runtimes are unsupported; existing images require separate upgrade and
acceptance before adopting this release. See the [SDK Python policy](README.md#python-compatibility).

## 1. External entrypoint

The benchmark image exposes one external command:

```bash
/root/startbenchmark.sh <suite> <checkpoint_source> [evaluation_num]
```

`evaluation_num` defaults to `50` and means the number of episodes evaluated for every selected task.

Supported suites are:

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

The command starts the user's SDK `Policy`. Runtime selectors, simulator commands, evaluator
commands, sockets, ports, K, and g are platform internals and are not part of the user contract.

### Checkpoint source

For an SDK-native user Policy:

- an `oss://...` source is copied recursively to `/root/benchmark/checkpoint`, and that directory is passed
  to `Policy.load()` as `context.checkpoint_path`;
- an absolute local file or directory is passed to `Policy.load()` unchanged;
- relative local paths and missing paths are rejected.

## 2. User Policy process contract

The user entrypoint installs `roboscope-kit`, implements `Policy`, and ends with:

```python
run(UserPolicy())
```

The evaluation image stores the user Policy process contract at the fixed path
`/workspace/roboscope-policy.json`:

```json
{
  "schema_version": 2,
  "command": [
    "/opt/user/bin/python",
    "/workspace/user_policy.py"
  ],
  "cwd": "/workspace"
}
```

The path is not configurable. `command` must be a non-empty string array whose executable is an absolute,
runnable file, and `cwd` must be an existing absolute directory. The quick test validates this file and
scored evaluation reloads and validates the fixed file from the submitted image before Policy startup.
Users do not configure path variables, sockets, ports, K, or g.

For every inference replica the platform:

1. starts the command with `ROBOSCOPE_INFERENCE_SOCKET`, `ROBOSCOPE_REPLICA_ID`, and the assigned
   `CUDA_VISIBLE_DEVICES`;
2. waits for the SDK Unix socket;
3. calls `Policy.load(SessionContext)` exactly once;
4. checks `PolicyCapabilities` before starting evaluator workers;
5. calls `Policy.reset(EpisodeContext)` before every episode;
6. calls `Policy.act(Observation)` whenever the worker needs a new action chunk;
7. after a successful reset, calls `Policy.end_episode(episode_id)` when the episode ends or aborts.

The Policy process owns mutable episode state and should key it by `episode_id`. The SDK does not store or
synchronize user state.

The `Policy.load()` RPC timeout inherits `BENCHMARK_READY_TIMEOUT`, which defaults to 900 seconds, so
model initialization uses the same configurable readiness budget as the LIBERO runtime.

Users do not declare evaluator concurrency. The platform starts isolated Policy replicas and chooses K/g
from measured end-to-end episode throughput and resource signals. SDK Policy autotune keeps one active
evaluator worker per isolated replica while ramping candidates, so temporal state never needs a
user-supplied concurrency limit.

## 3. Canonical LIBERO observation

The SDK-native LIBERO worker sends the following features:

| Feature | Type | Shape | Construction |
| --- | --- | --- | --- |
| `images.primary` | `uint8` | `(256, 256, 3)` | `agentview_image`, flipped on both image axes |
| `images.wrist` | `uint8` | `(256, 256, 3)` | `robot0_eye_in_hand_image`, flipped on both image axes |
| `state.proprioception` | `float32` | `(16,)` | EEF position 3 + EEF quaternion 4 + gripper qpos 2 + joint positions 7 |
| `language.instruction` | `str` | scalar | Official LIBERO task language instruction |

Every `Observation` also carries:

- `episode_id`: `task-<task_id:04d>-episode-<episode_index:04d>`;
- `step`: the zero-based policy action step within the episode;
- `timestamp_s`: wall-clock time when the request was constructed.

The server rejects undeclared features, missing required features, incorrect dtypes, and incorrect fixed
shapes before calling user inference code.

## 4. LIBERO action contract

The Policy returns `ActionChunk` containing either:

- one `float32` action with shape `(7,)`; or
- a `float32` chunk with shape `(N, 7)`, where `N` is declared in
  `PolicyCapabilities.supported_action_chunk_sizes`.

The seven values are the native LIBERO controller action: six end-effector delta values followed by the
gripper command. The SDK-native worker does not apply StarVLA response decoding, clipping, or normalization;
the Policy must return a benchmark-valid native action. Actions from a chunk are executed in order, one
simulator step at a time. A new observation is sent to the Policy only after the cached chunk is exhausted.

The SDK validates numeric dtype, action dimension, and declared chunk size before the worker receives the
action.

## 5. Episode protocol

Unless the platform selects an explicit subset through `BENCHMARK_TASK_IDS`, every task in the requested
suite is evaluated in ascending task-id order. Explicit task ids retain the order supplied by the platform.
`evaluation_num` episodes are created for each selected task, using episode indices `0..evaluation_num-1`.
The value must not exceed the official initial-state count for the selected tasks.

The LIBERO environment seed defaults to `7` and may be overridden by the platform through
`BENCHMARK_SEED`. It is applied when a task environment is created. User Policy code does not select the
benchmark seed.

For task `t` and episode index `e`:

1. the worker calls `Policy.reset()` with `task_id=str(t)` and the official instruction;
2. the environment is reset and initialized with `task_suite.get_task_init_states(t)[e]`;
3. the worker executes 10 native no-op settle actions `[0, 0, 0, 0, 0, 0, -1]`;
4. the Policy/environment loop runs until official success/termination or the suite horizon;
5. exactly one terminal `EvaluationRecord` is committed.

Suite horizons are:

| Suite | Maximum policy action steps |
| --- | ---: |
| `libero_spatial` | 220 |
| `libero_object` | 280 |
| `libero_goal` | 300 |
| `libero_10` | 520 |
| `libero_90` | 400 |

An episode is successful only when LIBERO returns `task_done=True`. `env.env.done` also terminates the
episode, but does not by itself convert a failed episode into a success. Reaching the horizon is a failure.

Terminal record identifiers are:

```text
task-<task_id:04d>-episode-<episode_index:04d>
```

The runtime rejects missing or duplicate terminal record identifiers. Official aggregation reports episode
and success counts per task; the standard result also reports total successes, failures, success rate,
steps, wall time, steps/s, and episodes/s.

## 6. Scheduling contract

The platform owns scheduling. Users do not set these values in inference code.

- `g` is the number of user Policy replicas.
- `K` is the number of persistent LIBERO evaluator workers.
- Work is stored in a durable filesystem queue.
- `BENCHMARK_CHUNK_SIZE` controls how many consecutive episodes of one task are committed as one work item;
  it defaults to `1`.
- A worker prefers more work from the same task so it can reuse the loaded environment.
- An uncommitted work item is requeued when a worker fails.
- Final accounting requires exactly one terminal record for every manifest episode.

All Policies use platform-owned end-to-end K/g probing for
scored evaluations. Candidates run real LIBERO episodes with isolated Policy replicas and paired K/g
growth. The platform selects the last candidate that improves episode throughput without approaching CPU
throttling, GPU utilization, or GPU-memory limits. Quick test remains fixed at `K=1/g=1`.

## 7. Product progress and result files

`/root/testlog.txt` always uses exactly three lines:

```text
用户模型权重下载进度 <0-100>%
测试套件下载进度 <0-100>%
LIBERO <suite> <total> <success> <failed> <status>
```

`status` is one of:

```text
idle
downloading
running
finish
error:<code>
```

The file is rewritten every five seconds while a run is active so the product can detect liveness. Its
`success` and `failed` fields are the authoritative product-facing episode counts.

Additional structured artifacts are written under `/root/testresult/autotune/`:

| Path | Meaning |
| --- | --- |
| `manifest.json` | Immutable benchmark metadata, work items, and expected record count |
| `progress.json` | Current completed episode counts |
| `result.json` | Runtime config, Policy capabilities, official result, standard result, resources, recovery, and K/g mode |
| `events.jsonl` | Ordered phase, progress, failure, recovery, and completion events |
| `errors.jsonl` | Append-only errors from Workers and the top-level evaluation runtime |
| `servers/` | Policy replica logs and optional inference metrics |
| `workers/` | Evaluator worker logs |
| `queue/` | Pending, running, and completed durable work records |
| `probe/metrics.json` | SDK Policy candidate measurements, rejection reasons, resource signals, and final K/g |
| `probe/events.jsonl` | Ordered SDK Policy candidate and selection events |

Videos are written to `/root/testvideo/` when video recording is enabled.

## 8. Error contract

Argument, environment, checkpoint, asset, runtime, and evaluation failures return a nonzero exit code and
write `error:<code>` to `/root/testlog.txt`. The runtime also appends structured error records to
`/root/testresult/errors.jsonl`; recoverable Worker failures remain in this journal even when the final
evaluation succeeds. Each Worker failure record includes the original traceback excerpt. If retries are
exhausted, the same excerpt is propagated through the top-level error for product log display. Common
wrapper exit codes are:

| Exit code | Meaning |
| ---: | --- |
| 2 | Missing or invalid wrapper arguments |
| 3 | LIBERO Python is not executable |
| 4 | Required benchmark runtime directory is missing |
| 5 | Self-test Python is not executable |
| 6 | Quick-test checkpoint or required OSS environment is invalid |

Runner failures use named codes such as `invalid_suite`, `invalid_evaluation_num`,
`invalid_checkpoint_source`, `local_checkpoint_missing`, `missing_autotune_runtime`, and
`evaluation_failed`.

## 9. Required validation sequence

### Hermetic protocol self-test

```bash
/root/startbenchmark.sh --self-test
```

This requires no OSS, GPU, LIBERO dataset, or real checkpoint. It starts an SDK Policy subprocess, loads
its capabilities, has two evaluator workers call it through `PolicyClient`, and verifies complete result
accounting.

### User quick test

```bash
/root/startbenchmark.sh --quick-test <suite> <oss_checkpoint_source>
```

Quick test fixes task id `0`, `K=1`, `g=1`, one episode, and disables video. It verifies checkpoint access,
Policy startup/loading, SDK communication, and a real LIBERO rollout. Policies require only the checkpoint
OSS environment.

### Scored evaluation

Run the normal entrypoint only after both checks pass. A model returning a correctly shaped action is not
enough: task membership, initial states, settle actions, horizon, success detection, terminal accounting,
and aggregation must all match this protocol.

## 10. Current support boundary

This document describes the completed SDK-native LIBERO protocol.
CALVIN has its own protocol document. LIBERO+, RoboTwin, RoboCasa, and CALVIN use the same platform-owned
Policy boundary, but their execution chains are not interchangeable: LIBERO+ reuses the LIBERO worker,
while RoboTwin, RoboCasa, and CALVIN have benchmark-specific workers. Each chain also requires its own
installed image manifest and external simulator, dataset, and pinned-environment artifacts. See the SDK
README's benchmark status table for the exact boundary; installing the PyPI package alone does not
provision any of these Benchmark runtimes.
