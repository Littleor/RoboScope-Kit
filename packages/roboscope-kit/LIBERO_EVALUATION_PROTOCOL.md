# LIBERO Evaluation Protocol

This document defines the SDK-native RoboScope LIBERO and LIBERO+ evaluation contract. The kit's benchmark
runtime owns benchmark enumeration, simulator processes, concurrency, recovery, and result aggregation.
User inference code owns only the public `Policy` lifecycle and model-specific preprocessing and
postprocessing.

The current protocol identifiers are:

- benchmark specification: `libero-v1` (LIBERO+: `libero-plus-v1`);
- action representation: `libero.delta-eef-v1`;
- SDK-native evaluator transport: `roboscope-policy-v1`.

`roboscope-kit` requires Python 3.11 or newer for the Policy and for the benchmark CLI. The LIBERO and
LIBERO+ simulator environments are CPython 3.10.20, the interpreter of their pinned simulator stack, and
their evaluator workers import the kit in that interpreter. Python 3.10 is outside the kit's
[supported range](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support);
the modules a LIBERO worker imports are kept importable on it, and the build scripts check that import,
but CI does not test it.

## 1. Environment and invocation

The LIBERO simulator is not a dependency of `roboscope-kit`. Build it with
[`benchmarks/libero/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero/build-runtime.sh) `<install-dir>`
(Linux x86_64, `conda` or `micromamba`, `git`, network access). The script:

1. checks out `Lifelong-Robot-Learning/LIBERO` at `8f1084e`, whose tree already contains the BDDL files,
   initial states, and assets;
2. creates a conda prefix from
   [`conda-specs.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero/conda-specs.txt)
   and installs the LIBERO source tree and then exactly
   [`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero/runtime-requirements.txt)
   (`robosuite==1.4.0`, `bddl==1.0.1`, `mujoco==3.2.3`, `numpy==1.24.4`, `torch==2.5.1+cpu`) without
   dependency resolution;
3. writes the LIBERO path config with the kit's `write_libero_config`, imports LIBERO and the kit's worker
   in the LIBERO interpreter as a worker does, and checks both resolve to the install;
4. copies
   [`benchmark-manifest.json`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero/benchmark-manifest.json)
   into `<install-dir>` and runs the kit's `--self-test` against it.

The kit side of steps 3 and 4 runs in `KIT_PYTHON` (default `python3`), which must import `roboscope_kit`
from a source tree, as described below. Every `runtime` path in the manifest is relative to the manifest
file:

| `runtime` key | Path under `<install-dir>` | Contents |
| --- | --- | --- |
| `python` | `env/bin/python` | the LIBERO interpreter that runs evaluator workers |
| `benchmark_root` | `LIBERO` | LIBERO sources; workers read `libero/libero/{bddl_files,init_files,assets}` |
| `config_path` | `config/config.yaml` | the LIBERO path config |

At evaluation time the adapter writes the same path config to `<output-dir>/worker-state/libero-config/`
and points every worker's `LIBERO_CONFIG_PATH` at it, so evaluation never reads or writes `~/.libero`. No
LIBERO demonstration dataset is needed. The conda prefix records absolute paths, so do not move
`<install-dir>` after the build.

Workers import the same `roboscope_kit` package the CLI imported, so `roboscope-kit` is not installed into
the LIBERO environment. The adapter puts a directory holding only that package on `PYTHONPATH`: a source
tree's `src`, or, for a pip install, a directory that links just the package, so the CLI's other
`site-packages` never shadow the LIBERO environment's. The build script performs the same import check.

Run an evaluation from that kit, in any supported Python:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark libero --suite libero_10 --checkpoint-source /models/my-checkpoint \
    --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/libero_10
```

Supported suites are:

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

`--episodes N` sets the number of episodes evaluated for every task; it defaults to the manifest's
`episodes`, 50. `--quick-test` runs a short real evaluation (section 9). Simulator commands, sockets,
ports, K, and g are runtime internals and are not part of the user contract.

### Checkpoint source

- An absolute local file or directory is passed to `Policy.load()` unchanged as
  `context.checkpoint_path`; a missing path is rejected.
- Relative paths and remote URLs are rejected. Download or otherwise prepare the checkpoint locally before
  running the evaluation.

## 2. User Policy process contract

The user entrypoint installs `roboscope-kit`, implements `Policy`, and ends with:

```python
if __name__ == "__main__":
    run(UserPolicy())
```

The runtime reads the Policy process contract from `--policy-config`:

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

`command` must be a non-empty string array whose executable is an absolute, runnable file, and `cwd` must
be an existing absolute directory. The runtime validates the file at the start of every run, before any
Policy starts.

For every inference replica the runtime:

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

Replica readiness and the `Policy.load()` RPC share the manifest's `ready_timeout_s`, 300 seconds in the
LIBERO and LIBERO+ manifests.

Users do not declare evaluator concurrency; the runtime chooses K and g (section 6).

## 3. Canonical LIBERO observation

The SDK-native LIBERO worker renders both cameras at 256 × 256 and sends the following features:

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
gripper command. The worker does not decode, clip, or normalize actions; the Policy must return a
benchmark-valid native action. Actions from a chunk are executed in order, one simulator step at a time.
A new observation is sent to the Policy only after the cached chunk is exhausted.

The SDK validates numeric dtype, action dimension, and declared chunk size before the worker receives the
action.

## 5. Episode protocol

A scored evaluation runs every task of the requested suite in ascending task-id order; a quick test runs
task `0` only. For every selected task, episodes use indices `0..episodes-1`. The value must not exceed the
50 initial states every official LIBERO task ships.

The environment seed is the manifest's `seed`, `7`. It is applied once when a worker creates a task
environment; a worker reuses that environment for consecutive episodes of the same task. User Policy code
does not select the benchmark seed. The manifest's `render_backend` (`egl` or `osmesa`) selects MuJoCo
rendering; with `egl`, each worker renders on its assigned GPU.

For task `t` and episode index `e`:

1. the worker calls `Policy.reset()` with `task_id=str(t)` and the official instruction;
2. the environment is reset and initialized with `task_suite.get_task_init_states(t)[e]`;
3. the worker executes the manifest's `settle_steps`, 10, native no-op actions `[0, 0, 0, 0, 0, 0, -1]`;
   if the episode terminates during them, it ends with zero policy steps;
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
steps, wall time, steps/s, and episodes/s. When every Worker completes a warm-up item and a later item
before the first Worker runs out of work, it also reports `steady_steps_per_s` and `steady_window`: the
throughput of all `K` Workers while every one is busy, without each Worker's first item.

`result.json` also records `item_timeline` (`completion_s`, the seconds from evaluation start at which the
controller first saw each work item completed, rounded up to its poll interval, and `worker_id` for each
item) and `record_success` (success of each terminal record identifier).

## 6. Scheduling contract

The runtime owns scheduling. Users do not set these values in inference code.

- `g` is the number of user Policy replicas.
- `K` is the number of persistent LIBERO evaluator workers.
- Work is stored in a durable filesystem queue.
- The manifest's `chunk_size`, default `1`, sets how many consecutive episodes of one task are committed as
  one work item.
- A worker prefers more work from the same task so it can reuse the loaded environment.
- An uncommitted work item is requeued when a worker fails.
- Final accounting requires exactly one terminal record for every manifest episode.

Scored evaluations choose K and g by running real LIBERO episodes with the Policy, as described in the
README's [automatic tuning](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#automatic-tuning-kg)
section; `--autotune-strategy` selects the search. Quick test is fixed at `K=1/g=1`. `--concurrency K
--replicas G` skips probing; the evaluation then runs at that fixed pair, and `result.json` records
`mode: fixed`. Tuning episodes never count toward the score.

Each probe runs the same representative task set, `probe_task_count` tasks (default 10, capped at the suite
size), with `probe_episodes_per_task` episodes per task (default 1). With `probe_items_per_worker` above 0,
a probe at `K` adds episodes per task until it holds at least that many work items per Worker, counting
`chunk_size` episodes per item. It defaults to 3 with the steady-state estimator, which the diagonal
strategy uses, and to 0 otherwise. Episodes per task never exceed the 50 initial states each LIBERO task
ships; a `K` that would need more is rejected as a Worker shortfall. Because only episodes per task grow,
every probe repeats the episodes of each smaller probe, and a probe's success rate is compared with the
baseline on the episodes both ran.

## 7. Output files

Every file is written under `--output-dir`:

| Path | Meaning |
| --- | --- |
| `result.json` | Runtime config, Policy I/O contract, official result, standard result, resources, recovery, `autotune` (selected `k`, `g`, and `mode`), `cuda_mps`, `item_timeline`, and `record_success` |
| `manifest.json` | Benchmark metadata, Policy I/O contract, work items, and expected record count |
| `progress.json` | Current completed episode counts |
| `metrics.json` | Phase timings: checkpoint preparation, autotune, Policy load, runtime preparation, and evaluation |
| `events.jsonl` | Ordered phase, progress, failure, recovery, and completion events |
| `errors.jsonl` | Append-only errors from Workers and the top-level evaluation runtime |
| `workers/` | Evaluator worker logs, one per worker attempt |
| `queue/` | Pending, running, and completed durable work records |
| `worker-state/` | Worker state files and the workers' LIBERO path config |
| `servers/` | Policy replica logs and optional inference metrics of a quick-test or fixed K/g run |
| `autotune/selection.json` | Candidate measurements, rejection reasons, resource signals, and the final K/g, updated as candidates start, advance, and stop |
| `autotune/policy-pool/` | Policy replica logs of an autotuned run; the same replicas serve the probes and the scored pass |
| `autotune/candidate-k<K>-g<G>/` | Each probe's own manifest, result, events, worker logs, and queue |
| `video/` | One MP4 per episode, `task-<TT>-episode-<EE>-<success\|failure>.mp4`, when the manifest sets `save_videos: true`; never written by a quick test |
| `self-test.json` | The runtime paths `--self-test` checked |

## 8. Error contract

Argument, manifest, checkpoint, Policy, runtime, and evaluation failures make the CLI exit nonzero. The
top-level failure is appended to `errors.jsonl` as an `evaluation-error` record with its `component`,
`error_type`, and `message`. Worker failures are appended to the same journal, with the original traceback
excerpt, even when the evaluation later succeeds. A failed worker's uncommitted work item is requeued and
the worker restarted; a worker that fails a second time for a reason other than running out of memory
aborts the evaluation, and the top-level error carries the excerpt of its final attempt. Each Policy replica
and worker also keeps its own log.

## 9. Required validation sequence

### Runtime self-test

```bash
python -m roboscope_kit.runtime.benchmark --benchmark libero --self-test \
    --benchmark-manifest <install-dir>/benchmark-manifest.json --output-dir /runs/libero-self-test
```

This needs no checkpoint, Policy, or GPU. It checks that the manifest supports LIBERO and that its
`python`, `benchmark_root`, and `config_path` exist, and writes `self-test.json`. `build-runtime.sh` runs
it as its last step.

### User quick test

```bash
python -m roboscope_kit.runtime.benchmark --benchmark libero --suite <suite> --quick-test \
    --checkpoint-source <checkpoint> --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json --output-dir /runs/libero-quick-test
```

Quick test fixes task id `0`, `K=1`, `g=1`, one episode, and disables video. It verifies checkpoint access,
Policy startup and loading, SDK communication, and a real LIBERO rollout.

### Scored evaluation

Run the scored evaluation only after both checks pass. A model returning a correctly shaped action is not
enough: task membership, initial states, settle actions, horizon, success detection, terminal accounting,
and aggregation must all match this protocol.

## 10. LIBERO+

LIBERO+ (`--benchmark libero_plus`, protocol `libero-plus-v1`) runs the LIBERO worker above against the
LIBERO-plus task suites, so sections 2–9 apply unchanged, with these differences:

| | LIBERO+ |
| --- | --- |
| Suites (tasks) | `libero_spatial` (2,402), `libero_object` (2,518), `libero_goal` (2,591), `libero_10` (2,519); no `libero_90` |
| Episodes per task | manifest `episodes`, 1 |
| Probe workload | `probe_task_count` tasks × `probe_episodes_per_task` episodes, fixed for every `K` |

Build it with
[`benchmarks/libero_plus/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_plus/build-runtime.sh) `<install-dir>`
(Linux x86_64, `conda` or `micromamba`, `git`, `curl`, `unzip`, network access). It checks out
`sylvestf/LIBERO-plus` at `4976dc3`, downloads `assets.zip` (6.4 GB) from the `Sylvest/LIBERO-plus` Hugging
Face dataset at revision `dd2bd61`, verifies its SHA-256
`96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`, and moves its `assets` directory to
`LIBERO-plus/libero/libero/assets`. Set `LIBERO_PLUS_ASSETS_ZIP` to an already downloaded `assets.zip` to
skip the download; the checksum still applies. The environment is the recovered LIBERO+ set in
[`conda-specs.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_plus/conda-specs.txt)
and
[`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_plus/runtime-requirements.txt).
The remaining steps and the manifest layout match section 1, with `benchmark_root` set to `LIBERO-plus`.

## 11. Other benchmarks

CALVIN and MetaWorld have their own protocol documents. RoboTwin, RoboCasa, CALVIN, and MetaWorld use the
same Policy boundary, but have benchmark-specific workers, manifests, and simulator environments. See the
README's
[supported benchmarks](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#supported-benchmarks)
table; installing the PyPI package alone does not provision any benchmark environment.
