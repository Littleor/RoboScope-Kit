# LIBERO Evaluation Protocol

This document defines the SDK-native RoboScope LIBERO, LIBERO+, and LIBERO-Pro evaluation contract. The kit's benchmark
runtime owns benchmark enumeration, simulator processes, concurrency, recovery, and result aggregation.
User inference code owns only the public `Policy` lifecycle and model-specific preprocessing and
postprocessing.

The current protocol identifiers are:

- benchmark specification: `libero-v1` (LIBERO+: `libero-plus-v1`; LIBERO-Pro: `libero-pro-v1`);
- action representation: `libero.delta-eef-v1`;
- SDK-native evaluator transport: `roboscope-policy-v1`.

`roboscope-kit` requires Python 3.11 or newer for the Policy and for the benchmark CLI. The LIBERO and
LIBERO+ simulator environments are CPython 3.10.20, the interpreter of their pinned simulator stack, and
their evaluator workers import the kit in that interpreter. Python 3.10 is outside the kit's
[supported range](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support);
the modules a LIBERO worker imports are kept importable on it, and the build scripts check that import,
but CI does not test it. The LIBERO-Pro environment is CPython 3.11.18 (section 11).

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
and worker also keeps its own log. Output-directory rejections are reported to stderr
without modifying the previous run's logs.

## 9. Required validation sequence

Use a new or empty output directory for each run. A directory containing only `errors.jsonl` can be
retried; previous errors are preserved. Other run artifacts require a new directory. Keep shell and
scheduler logs outside the output directory. The examples below allocate a fresh directory each time.

### Runtime self-test

```bash
SELF_TEST_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/libero-self-test.XXXXXX")
python -m roboscope_kit.runtime.benchmark --benchmark libero --self-test \
    --benchmark-manifest <install-dir>/benchmark-manifest.json --output-dir "$SELF_TEST_OUTPUT"
```

This needs no checkpoint, Policy, or GPU. It checks that the manifest supports LIBERO and that its
`python`, `benchmark_root`, and `config_path` exist, and writes `self-test.json`. `build-runtime.sh` runs
it as its last step.

### User quick test

```bash
QUICK_TEST_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/libero-quick-test.XXXXXX")
python -m roboscope_kit.runtime.benchmark --benchmark libero --suite <suite> --quick-test \
    --checkpoint-source <checkpoint> --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json --output-dir "$QUICK_TEST_OUTPUT"
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

## 11. LIBERO-Pro

LIBERO-Pro (`--benchmark libero_pro`, protocol `libero-pro-v1`) runs the LIBERO worker above against the
released single-perturbation suites of [LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO). Sections 2–9
apply, with the differences below.

### Pinned inputs

| Input | Pin |
| --- | --- |
| Source, assets, base suites | `Zxy-MLlab/LIBERO-PRO` at `eafdb809426b13153aa1e4c42d6601844217dfec` (2026-03-23, head of master); its tree carries `libero/libero/assets` and the base LIBERO BDDL and initial-state files |
| Perturbed task files | Hugging Face dataset `zhouxueyang/LIBERO-Pro` at revision `c86fc3b8293185a6f373677018ff3e37f8391602` (CC-BY-4.0): 160 BDDL and 160 `.pruned_init` files, each checked against [`task-files.sha256`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_pro/task-files.sha256) |
| Task registry | the source revision's `libero/libero/benchmark/__init__.py` and `libero_suite_task_map.py` (the parallel-evaluation registry upstream added on 2025-11-05) |
| Interpreter | CPython 3.11.18 from conda-forge, [`conda-specs.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_pro/conda-specs.txt) |
| Packages | [`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_pro/runtime-requirements.txt): the LIBERO lock (`robosuite==1.4.0`, `bddl==1.0.1`, `mujoco==3.2.3`, `numpy==1.24.4`, `torch==2.5.1+cpu`) with `matplotlib` 3.5.3 → 3.7.5 (+ `contourpy` 1.3.2), the only change CPython 3.11 needs, and the `exceptiongroup`/`tomli` backports dropped |

Upstream documents Python 3.8.13 with torch 1.11; this runtime is the kit's minimum supported interpreter
instead, verified by the build's import checks, the suite check below, and the evaluations in this section.
`evaluation_config.yaml` and `perturbation.py` are not used: the released files are already the
single-perturbation configurations they would generate.

Build it with
[`benchmarks/libero_pro/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/libero_pro/build-runtime.sh) `<install-dir>`
(Linux x86_64, `conda` or `micromamba`, `git`, `curl`, network access, about 2 GB of disk). It fetches the
pinned commit only, downloads the 320 task files from the pinned dataset revision into
`LIBERO-PRO/libero/libero/{bddl_files,init_files}/<suite>/`, verifies every checksum, creates the
environment, and then, besides the import checks of section 1, loads every supported suite from the
LIBERO-Pro registry and checks it has 10 tasks, a BDDL file per task, and 50 initial states per task.
`LIBERO_PRO_URL` and `LIBERO_PRO_FILES_URL` select mirrors; the commit and checksums still apply. The
manifest layout matches section 1 with `benchmark_root` set to `LIBERO-PRO`. `--self-test` additionally
checks that each suite has its 10 BDDL and 10 initial-state files.

Evaluate one suite per run, for example with the StarVLA demo snapshot:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark libero_pro --suite libero_goal_task --checkpoint-source /models/starvla-libero \
    --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/RoboScope-Kit/examples/starvla/roboscope-policy.json \
    --output-dir /runs/libero_pro/libero_goal_task
```

### Supported suites

Each suite is one base LIBERO suite under one perturbation dimension. Suite names are upstream's.

| Dimension (upstream flag) | Suffix | What changes | Suites |
| --- | --- | --- | --- |
| Object (`use_object`) | `_object` | object appearance, color, and scale | `libero_goal_object`, `libero_spatial_object`, `libero_10_object`, `libero_object_object` |
| Position (`use_swap`) | `_swap` | initial object positions (swapped regions) | `libero_goal_swap`, `libero_spatial_swap`, `libero_10_swap`, `libero_object_swap` |
| Semantic (`use_language`) | `_lan` | paraphrased instruction | `libero_goal_lan`, `libero_spatial_lan`, `libero_10_lan`, `libero_object_lan` |
| Task (`use_task`) | `_task` | new instruction and goal in the same scene | `libero_goal_task`, `libero_spatial_task`, `libero_10_task`, `libero_object_task` |

Every suite has the 10 tasks of its base suite, in the registry's order (the base suite's task order), and
every task ships 50 initial states, so a scored suite is 10 tasks × 50 episodes = 500 episodes, episode `e`
using initial state `e`. The standard single-perturbation profile is all 16 suites, 8,000 episodes. Each
suite runs at its base suite's horizon, as upstream's `TASK_MAX_STEPS` does: `libero_goal_*` 300,
`libero_spatial_*` 220, `libero_10_*` 520, `libero_object_*` 280 policy steps. Seed (7), settle steps
(10), observations, actions, success, and termination are those of sections 3–5.

One run evaluates one suite, and its official result is that suite's per-task table. Report every suite,
that is every (base suite, dimension) cell, on its own. A per-dimension or overall figure is a mean of
suite success rates and must be labeled as such; never pool LIBERO-Pro suites with each other or with
plain LIBERO into one success count, and never combine different dimensions into one score.
`result.json` metadata records `base_suite`, `perturbation`, `dimension`, `max_steps`, and
`instruction_source` for every run.

Not supported, and rejected as suites:

- **Environment** (`*_env`): no files were released. The maintainers withheld them for simulation
  stability issues ([LIBERO-PRO#9](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/9)); upstream's README
  notes that replacing the environment can make objects on the table move, and
  [LIBERO-PRO#24](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/24) shows a misplaced initial layout.
  Generating them locally is not frozen either (next item), so no result here covers Environment.
- **Combined perturbations** (`*_temp`): upstream generates them at evaluation time with `perturbation.py`
  into shared `*_temp` directories keyed by a `log.txt` of the flags, so concurrent runs overwrite each
  other's files, and the generation seed falls back to `configs.get("seed", int)` (the `int` type), which
  is not a reproducible seed. Upstream's rule that the Task perturbation cannot be combined with others
  would apply to any future combined profile, which must be frozen and reported separately from this one.
- **The 2026-07 seven-case extension** of the dataset (`bddl_files/0*_*`, 40 tasks × 7 cases): its
  `:perturbation_config` fields need a LIBERO-Pro-aware parser and evaluation integration that the pinned
  source revision does not contain, so the build does not download it.

### Instruction source

`language.instruction` is the `(:language ...)` of the task's perturbed BDDL file as the environment parsed
it (`env.language_instruction`, lowercased and whitespace-normalized by the BDDL tokenizer), not
`Task.language`. The released files keep the original BDDL file names, and upstream's registry derives
`Task.language` from the file name, so `Task.language` is the unperturbed LIBERO instruction: feeding it
would silently undo the Semantic and Task perturbations (for example, `libero_goal_task`'s
`put_the_bowl_on_the_plate` asks for "put the wine bottle on the plate"). The maintainers prescribe
`env.language_instruction` ([LIBERO-PRO#14](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/14));
[LIBERO-PRO#27](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/27) measures how much the choice moves Task
scores. The worker selects this source with `--instruction-source bddl`; plain LIBERO and LIBERO+ keep
`Task.language`.

This also differs from plain LIBERO where the dimension does not perturb language. The BDDL wording of
the unperturbed base tasks is not their file-name instruction for `libero_spatial` ("pick the akita black
bowl ..." for "pick up the black bowl ..."), `libero_object` ("pick the ..." for "pick up the ..."), and 5
of 10 `libero_goal` tasks, so the Object and Position suites of those bases also see that wording; for
`libero_10` the two are identical. Runners that feed `Task.language` score something else: StarVLA's
LIBERO client does (checked at `7e4a473`), and LIBERO-PRO#27 reports the same for the openvla-oft, openpi,
and lerobot clients. Scores produced that way, possibly including published ones, are not comparable to
these for the Semantic and Task dimensions.

Known upstream content issues that this protocol does not correct, because the released files are scored
as released: [LIBERO-PRO#38](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/38) (`libero_10_task`
KITCHEN_SCENE8 left/right naming) and
[LIBERO-PRO#25](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/25) (`libero_object_swap` swaps the target
with a distractor, not with the basket). The unbalanced goals of
[LIBERO-PRO#28](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/28) are not in the released files.

### Correctness baseline

Recorded on 2026-11-05 with this change's kit (0.8.0b29), the runtime built by this section's script, the
[StarVLA demo](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/examples/starvla/libero_pro/demo.yaml)
(`StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1` at `a60fff4`, StarVLA `7e4a473`, bf16, `starvla_policy.py`), and one
RTX 3080 10 GB.

The native reference is StarVLA `7e4a473`'s own evaluator: `deployment/model_server/server_policy.py
--use_bf16` serving the same checkpoint, and its LIBERO client `examples/LIBERO/eval_files/eval_libero.py`,
run in the LIBERO-Pro interpreter with only the two changes LIBERO-Pro prescribes for such a client:
perturbed suites take their base suite's horizon (upstream README `TASK_MAX_STEPS`) and the instruction is
`env.language_instruction` (LIBERO-PRO#14). The harness also writes one line per episode and skips videos.
The Kit ran the same Policy, wrapped only to log every request, at fixed K=2/g=1. Both sides used seed 7
and initial states 0–4 of every task of `libero_goal_task`, `libero_spatial_lan`, `libero_object_swap`, and
`libero_10_object` (one suite per dimension, 200 paired episodes):

| Paired check | Equal |
| --- | ---: |
| success | 200 / 200 |
| policy steps executed | 197 / 200 |
| instruction | 200 / 200 |
| first agent-view image (SHA-256 of the flipped 256 × 256 frame) | 194 / 200 |
| first action chunk, bit for bit, where the first image is equal | 194 / 194 |
| whole executed action sequence, bit for bit | 186 / 200 |

Per suite, the Kit and native success counts are 4/4, 48/48, 0/0, and 23/23 of 50. What these 200
episodes verify, and what they do not:

- Instruction: identical in 200 of 200 episodes.
- Observation and action mapping: in the 194 episodes whose first agent-view frame was identical on both
  sides, the first 8-action chunk the environment executed was also identical bit for bit. On those starts,
  the image flip, resizing, instruction, model input, unnormalization, gripper mapping, and chunk order give
  the same first actions. Only the primary frame and the actions were hashed, not the wrist frame.
- Whole episodes: the full executed action sequence was identical in 186 of 200, the policy step count in
  197 of 200, and success in 200 of 200.

The differences, and what explains them:

- 6 episodes start from a different first frame. LIBERO seeds a task environment once and reuses it, and
  the state after `reset()`, `set_init_state()`, and the settle steps still depends on the random draws of
  that environment's earlier resets. A CPU-only check on one task (`libero_spatial_lan` task 9) showed this:
  a fresh environment and one that had already run episode 0 reach different simulator states for episode 1,
  and re-seeding before the episode restores the fresh state. The native client always runs a task's
  episodes in order on one environment. With K=2 the Kit split two tasks between Workers, and the 6 episodes
  are exactly those whose Worker environment had a different history (`libero_goal_task` task 9 episodes
  3–4, `libero_spatial_lan` task 9 episodes 1–4). The 3 step-count differences are among them: successes
  1–3 steps apart. This is upstream LIBERO behavior that plain LIBERO shares. The Kit keeps upstream's
  seed-once semantics; re-seeding per episode would change outcomes relative to upstream.
- 8 episodes start identically but diverge later in the action sequence, with the same success and step
  count. This was not traced; it is consistent with GPU inference that is not bitwise reproducible, but that
  was not shown.

This is one run per side on 4 of the 16 suites and initial states 0–4. It supports the mappings above; it
does not establish statistical equivalence of scores, or anything about the suites and episodes it did not
cover.

### Full evaluation

All 16 suites of the standard profile, 500 episodes each (10 tasks × initial states 0–49, every episode
exactly once), 8,000 episodes in total, with the same demo Policy and snapshot as above. 13 suites ran on
the RTX 3080 10 GB at fixed K=2/g=1. `libero_10_swap`, `libero_10_lan`, and `libero_10_task` ran on one
RTX 4090 48 GB at fixed K=8/g=2 with `OMP_NUM_THREADS=8`, sharing the GPU with other evaluation jobs, using
a runtime built by the same script (identical package set). Success rates (each cell one suite; the last
row is a mean of suite success rates):

| Base suite | Object | Position | Semantic | Task |
| --- | ---: | ---: | ---: | ---: |
| `libero_goal` | 70.2% | 1.2% | 88.6% | 9.4% |
| `libero_spatial` | 95.2% | 0.0% | 93.4% | 55.0% |
| `libero_10` | 47.4% | 0.0% | 91.8% | 10.6% |
| `libero_object` | 90.2% | 0.0% | 96.4% | 9.6% |
| Mean of the 4 suites | 75.8% | 0.3% | 92.5% | 21.1% |

| Suite | Episodes | Successes | Hardware, K/g |
| --- | ---: | ---: | --- |
| `libero_goal_object` | 500 | 351 | 3080, 2/1 |
| `libero_goal_swap` | 500 | 6 | 3080, 2/1 |
| `libero_goal_lan` | 500 | 443 | 3080, 2/1 |
| `libero_goal_task` | 500 | 47 | 3080, 2/1 |
| `libero_spatial_object` | 500 | 476 | 3080, 2/1 |
| `libero_spatial_swap` | 500 | 0 | 3080, 2/1 |
| `libero_spatial_lan` | 500 | 467 | 3080, 2/1 |
| `libero_spatial_task` | 500 | 275 | 3080, 2/1 |
| `libero_10_object` | 500 | 237 | 3080, 2/1 |
| `libero_10_swap` | 500 | 0 | 4090, 8/2 |
| `libero_10_lan` | 500 | 459 | 4090, 8/2 |
| `libero_10_task` | 500 | 53 | 4090, 8/2 |
| `libero_object_object` | 500 | 451 | 3080, 2/1 |
| `libero_object_swap` | 500 | 0 | 3080, 2/1 |
| `libero_object_lan` | 500 | 482 | 3080, 2/1 |
| `libero_object_task` | 500 | 48 | 3080, 2/1 |

### Hardware and performance baseline

On this 10 GB GPU, one bf16 replica of the 3B checkpoint held 8.3–8.8 GB and one EGL Worker about 0.46 GB
of the same GPU (measured with a single Worker). At K=2/g=1, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the Policy's environment, the peak over the 13 suites run
on it was 9,832 MiB of the card's 10,240 MiB. In the one K=4/g=1 run, the Policy ran out of memory and the run
aborted. K=3 was not tried: another Worker's 0.46 GB is not expected to fit. Autotune was not used, because
its probes would try larger K. The RTX 3080 runs use fixed `--concurrency 2 --replicas 1`. A larger GPU
raises this limit (the RTX 4090 runs below use K=8/g=2); rendering on another GPU was not tested.

| `libero_goal_object`, 500 episodes, K=2/g=1 | Value |
| --- | ---: |
| Policy load | 16.8 s |
| Runtime preparation | 1.6 s |
| Evaluation | 2,380 s |
| Episodes/s | 0.210 |
| Steps/s (steady) | 33.9 (34.0) |
| Inference wait / episode time | 35% |
| GPU utilization mean / max | 31% / 75% |
| Peak GPU memory | 9.81 GB |

A Quick Test (`libero_goal_task`) takes 40.7 s end to end, of which Policy load 18.8 s and the 300-step
episode 18.3 s. A clean build takes about 21 minutes on a slow shared link, almost all of it downloads.

On the shared RTX 4090 48 GB (2 × Xeon Platinum 8470Q host, Kit and Policy pinned to cores 0–63), the three `libero_10` suites ran at K=8/g=2: `libero_10_swap` 102 steps/s (2,541 s), `libero_10_lan` 131 steps/s (1,039 s), `libero_10_task` 147 steps/s (1,717 s). Two bf16 replicas held about 8.5 GB each. The GPU was busy with other jobs at the same time (device utilization near 100%, inference wait about 70% of episode time), so these numbers are not comparable with the RTX 3080 runs; `result.json`'s GPU memory and utilization cover the whole device, other jobs included. Limit the Policy's CPU threads on many-core hosts: in two short K=6/g=2 runs on this host, the Policy without `OMP_NUM_THREADS` kept about 12 cores busy per process and the run reached about 15 steps/s; with `OMP_NUM_THREADS=8` it reached 86 steps/s. The other jobs' load was not controlled between the two runs.

## 12. Other benchmarks

CALVIN, MetaWorld, and robomimic have their own protocol documents. RoboTwin, RoboCasa, CALVIN, MetaWorld,
and robomimic use the
same Policy boundary, but have benchmark-specific workers, manifests, and simulator environments. See the
README's
[supported benchmarks](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#supported-benchmarks)
table; installing the PyPI package alone does not provision any benchmark environment.
