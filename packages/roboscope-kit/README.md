# RoboScope Kit

[![MINT Lab@SJTU](https://img.shields.io/badge/MINT%20Lab%40SJTU-1565c0)](https://mint-sjtu.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.03850-b31b1b.svg)](https://arxiv.org/abs/2611.03850)
[![PyPI](https://img.shields.io/pypi/v/roboscope-kit)](https://pypi.org/project/roboscope-kit/)

`roboscope-kit` evaluates your model on standard robot simulation benchmarks. You implement one `Policy`
class; the benchmark runtime starts your Policy, drives the simulators, tunes the evaluation's concurrency,
and scores the results. The design and measurements are in the
[technical report](https://arxiv.org/abs/2611.03850).

```bash
python -m pip install roboscope-kit
```

Python 3.11 or newer is required. The import name is `roboscope_kit`.

> Versions up to `0.7.0b77` were published as `roboscope-inference-sdk` and imported as
> `roboscope_inference_sdk`. From `0.8.0b1` on, use `roboscope-kit` and `roboscope_kit`; no alias is
> provided.

## Quick start

```python
import numpy as np

from roboscope_kit import ActionChunk, BenchmarkName, Policy, PolicyCapabilities, run


class UserPolicy(Policy):
    def load(self, context):
        self.model = load_model(context.checkpoint_path)
        return PolicyCapabilities(
            supported_benchmarks=(BenchmarkName.LIBERO,),
            required_features=("images.primary", "state.proprioception"),
            supported_suites={BenchmarkName.LIBERO: ("libero_10",)},
            supported_action_chunk_sizes=(1,),
        )

    def act(self, observation):
        return ActionChunk(np.asarray(self.model.predict(observation.features), dtype=np.float32))


if __name__ == "__main__":
    run(UserPolicy())
```

The runtime starts this script once per inference replica. It passes the staged checkpoint and the
benchmark contract to `Policy.load()`, then calls `act()` through a local socket. Your process never starts
a simulator or chooses how many evaluators run.

## How evaluation works

An evaluation takes two files:

- the **Policy process contract** (`--policy-config`), which tells the runtime how to start your Policy:

  ```json
  {
    "schema_version": 2,
    "command": ["/opt/my-policy/bin/python", "/opt/my-policy/user_policy.py"],
    "cwd": "/opt/my-policy"
  }
  ```

- the **benchmark manifest** (`--benchmark-manifest`, a `benchmark-manifest.json`), which lists the
  benchmark's suites and protocol, points `runtime` at the benchmark's Python environment, simulator source,
  and assets, and sets the evaluation defaults. Relative runtime paths resolve against the manifest's
  directory.

The simulators are not dependencies of this package. Each benchmark directory under
[`benchmarks/`](https://github.com/MINT-SJTU/RoboScope-Kit/tree/main/benchmarks) has a `build-runtime.sh` that builds the pinned simulator
environment into a directory you choose and copies the benchmark manifest there:

```bash
benchmarks/libero/build-runtime.sh /sim/libero
```

Run the evaluation with the benchmark CLI:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark libero --suite libero_10 --checkpoint-source /models/my-checkpoint \
    --benchmark-manifest /sim/libero/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/libero_10
```

`--checkpoint-source` is an absolute path, passed to `Policy.load()` unchanged, or an `oss://` prefix, which
is downloaded into `--checkpoint-dir`. The runtime validates the Policy process contract before every run.
`--quick-test` runs a short real evaluation with one Worker and one Server; without it, the run is a full
scored evaluation. `--self-test` only checks that the manifest's runtime paths exist. Results, progress,
and logs are written under `--output-dir`.

The runtime is built on three ideas:

- **Server**: a process that loads one replica of your `Policy` and answers action requests over a local
  socket. Several Workers can share one Server, which can batch their requests. Autotune calls the number of
  Servers `g`.
- **Worker**: a process that owns one simulator. It claims Work Items, runs their episodes, sends
  observations to a Server, and applies the returned actions. It never loads your model. Autotune calls the
  number of Workers `K`.
- **Work Item**: the unit of benchmark work, claimed atomically from a durable queue. Its size depends on
  the benchmark; a RoboTwin Work Item, for example, is a window of episodes from one task. If a Worker fails,
  its Work Item goes back to the queue; a finished Work Item is recorded once and can never be scored twice.

### Automatic tuning (K/g)

The runtime picks `K` and `g` for you by running real episodes with your Policy. Every candidate runs episodes
from the same representative task set. The smallest safe `K` sets a success-rate baseline; a faster
candidate is accepted only if it keeps at least 85% of that baseline and stays within resource limits.
Candidates that lose Workers, run out of memory, or leave episodes unfinished are rejected.

Each candidate is limited to 15 minutes and the whole search to 30 minutes, after which the last healthy
pair is used. Progress is written to `<output-dir>/autotune/selection.json` as candidates start,
advance, and stop. Tuning episodes never count toward the score.

The default search is `diagonal`, described below, on every benchmark. `two-phase`, which ramps `K` on one
replica per GPU and then ramps replicas when GPU utilization shows inference is the bottleneck, remains
available. `--autotune-strategy <name>` selects a search for one evaluation, and `autotune_strategy: <name>`
in a benchmark manifest selects it for every evaluation; the flag wins.

`diagonal` doubles `K` on one replica per GPU until a doubling raises throughput by less than 25% (a step
the CPU cap shortens needs the same share of linear scaling); the accepted `K` per replica is how many
Workers one replica keeps busy. It then adds one replica per GPU at a time with `K` scaled by that ratio,
keeping each step that raises throughput by at least 3% and skipping a step whose busiest GPU,
extrapolated from the measured replica footprint and the marginal memory per Worker, would not fit. It
compares steady-state throughput: each probe gives every Worker three work items, drops each Worker's first
item, and stops measuring when the first Worker runs out of work. A probe grows with `K` by adding episodes,
sequences, or tasks to a nested set, so it repeats every episode of a smaller probe. Where the catalog runs
out (RoboCasa's 24 tasks, MetaWorld's 50), larger `K` cannot be measured and the search keeps the last
accepted configuration. Probes it measures but does not keep are
marked `rejected` in `selection.json`.

To measure or reproduce one configuration, add `--concurrency K --replicas G`, for example
`--concurrency 12 --replicas 3`. The evaluation then runs at that `K` and `g` without probing, and
`result.json` records `"autotune": {"k": K, "g": G, "mode": "fixed"}`. Both flags are required together,
and neither can be combined with `--quick-test` or `--autotune-strategy`.

### Progress and errors

RoboTwin and RoboCasa report each finished episode as it completes, so progress moves while a long Work
Item is still running; the official score still uses only each item's final result. Progress is written to
`<output-dir>/progress.json` and `<output-dir>/events.jsonl`.

All runtime errors are appended to `<output-dir>/errors.jsonl`: recoverable Worker failures as well as
checkpoint, Policy Server, simulator, and configuration errors. When a RoboTwin simulator crashes with a
normal error exit, the episodes it finished are kept and the rest of that window count as failures, so the
evaluation continues. Out-of-memory and signal exits restart the Worker and replay its Work Item, with one
retry per run. Each component also keeps its own detailed log.

## Writing a Policy

### Capabilities and action contracts

`load()` returns `PolicyCapabilities`, which declares the benchmarks and suites you support, the
observation features you need, the action-chunk sizes that are safe for your model, and the maximum
inference batch size. It can also choose an action representation and an observation stride per benchmark:

```python
PolicyCapabilities(
    supported_benchmarks=(BenchmarkName.ROBOTWIN,),
    required_features=("images.head", "state.joints"),
    supported_action_chunk_sizes=(37,),
    action_representations={BenchmarkName.ROBOTWIN: "robotwin.ee-compact-quat-v1"},
    observation_strides={BenchmarkName.ROBOTWIN: 1},
)
```

The runtime checks each declared representation against the benchmark specification and requires every
replica to agree on the same action contract before evaluation starts. If you omit
`action_representations`, the benchmark default is used. The chosen contract and where it came from are
recorded in the inference metadata.

RoboTwin accepts a positive observation stride. The default is 1, so every new action chunk gets a fresh
observation. Randomized lighting, the viewer, video logging, and point-cloud observations always render
every step, whatever stride you declare.

Concurrency (`K`) and replica count (`g`) are not capabilities; the runtime chooses them, or you fix them
with `--concurrency` and `--replicas` (see [Automatic tuning](#automatic-tuning-kg)).

### Episode state

Before each episode the runtime calls `reset(EpisodeContext)`; every observation carries its `episode_id`;
when the episode ends it calls `end_episode(episode_id)`. The SDK forwards these calls but does not store
Policy state for you. If your Policy keeps state, key it by `episode_id` and release it in `end_episode()`.
The [X-VLA example](https://github.com/MINT-SJTU/RoboScope-Kit/tree/main/examples/xvla) is a stateful
Policy; the [StarVLA example](https://github.com/MINT-SJTU/RoboScope-Kit/tree/main/examples/starvla) is
stateless.

For RoboDojo, a Policy that needs observations between actions in a chunk can declare
`step_observation_updates=("robodojo",)` in `PolicyCapabilities` and implement `observe(Observation)`.
The observation passed to `act()` starts each chunk; `observe()` receives the intermediate steps in order.
Updates are synchronous by default. The RoboDojo manifest can opt into an experimental bounded queue;
updates are drained before the next `act()` and before ending the episode.

### Input preprocessing

Each benchmark sends its canonical observation unchanged: no model-specific resizing, normalization, or
channel/layout conversion. Your `Policy` does those transformations before calling the model, so model
input sizes are not part of the benchmark or Policy API.

## Supported benchmarks

Each benchmark has an execution chain in this package. The simulators, datasets, and assets are not
dependencies of the PyPI package; the benchmark manifest points the runtime at them.

| Benchmark | Execution chain | Environment the manifest points at |
| --- | --- | --- |
| LIBERO | SDK-native adapter and persistent LIBERO worker | Pinned LIBERO environment, simulator source, dataset, and initial states |
| LIBERO-Plus | SDK-native adapter reusing the LIBERO worker | Pinned LIBERO-Plus environment, simulator source, dataset, and initial states |
| RoboTwin | SDK-native adapter and worker calling the pinned RoboTwin launcher | RoboTwin environment and source, deploy policy, checkpoint assets, and validated seed cache |
| RoboDojo | SDK-native adapter and worker calling the pinned RoboDojo simulator | Isaac Sim environment, RoboDojo source and evaluation assets |
| RoboCasa | SDK-native adapter and persistent RoboCasa worker | Pinned RoboCasa environment, simulator and task source, and dataset assets |
| CALVIN | SDK-native adapter and persistent CALVIN worker | Pinned CALVIN environment, benchmark and config source, dataset, and evaluation-sequence list |
| MetaWorld | SDK-native adapter and persistent MetaWorld MT50 worker | Pinned MetaWorld/MuJoCo environment, task assets, and the frozen MT50 (seed 42) train-goal artifact |

LIBERO and CALVIN convert native observations to the canonical feature names and send native 7-dimensional
actions back to their simulators. CALVIN's task-D-to-task-D path runs the official 1,000 long-horizon
sequences with static and gripper images, proprioception, and language. The suites and runtime paths of
each benchmark are defined by its benchmark manifest.

The exact contracts are in the protocol documents:

- [LIBERO](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md):
  environment build, invocation, checkpoint staging, observation/action mapping, episode setup and horizons,
  success detection, output files, errors, and the validation sequence; also LIBERO-Plus.
- [CALVIN](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/CALVIN_EVALUATION_PROTOCOL.md):
  manifest, observation/action mapping, official long-horizon evaluator, aggregation, and validation.
- [MetaWorld](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/METAWORLD_EVALUATION_PROTOCOL.md):
  MT50 manifest, camera setup, official training-goal evaluation, four-difficulty macro aggregation, and
  validation.
- [RoboDojo](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robodojo/README.md):
  environment build, suites, observation and action layouts, step observation updates, and experimental
  runtime switches.
- [RoboTwin fixed inputs](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robotwin/SEED_PROTOCOL.md):
  since 0.7.0b51, the manifest freezes RoboTwin's 50 × 100 inputs per suite under
  `robotwin2-platform-native-step-lim-v2`. The clean entries are unchanged and the randomized pool is new,
  so report randomized v1 and v2 scores separately.

Integration, autotune, and throughput measurements are in the
[RoboScope Kit technical report](https://arxiv.org/abs/2611.03850).

## CUDA MPS (opt-in)

`--cuda-mps` runs the Policy replicas under NVIDIA CUDA MPS, so replicas sharing a GPU run their kernels
concurrently instead of taking turns. It is off by default. With it, the runtime:

- starts a private `nvidia-cuda-mps-control` daemon for the visible GPUs with 100% active-thread access, and
  refuses to start if another MPS daemon or server is already running;
- attaches every Policy replica to the daemon and fails the run if a replica is not an MPS client
  (simulator Workers are not attached);
- runs autotune and the scored pass in the same mode, so `K`/`g` are tuned for MPS;
- stops a replica's MPS clients through the daemon before stopping or rebuilding it, and shuts the daemon
  down when the run ends;
- writes daemon logs to `<output-dir>/cuda-mps/` and records the session under `cuda_mps` in `result.json`
  (`{"enabled": false}` when off).

The GPU must be in Default or Exclusive-Process compute mode, and the host must provide
`nvidia-cuda-mps-control`. Replicas pick their GPU by ordinal within the daemon's devices, so a Policy must
not change `CUDA_VISIBLE_DEVICES`.

## Cross-instance sharding (Beta)

Sharding splits one evaluation across several independent GPU machines. It is in Beta; evaluations are not
sharded unless you pass `--shard-index`.

Every instance builds the same full list of Work Items and keeps the items whose position satisfies
`position % N == shard_index`. No coordination is needed, and RoboTwin seeds stay identical to a
single-instance run. Run one instance per shard:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark robotwin --suite demo_clean --checkpoint-source ... \
    --benchmark-manifest ... --policy-config ... \
    --shard-index I --shard-count N --output-dir /runs/shard-I
```

Without `--shard-count`, the count is derived from the workload (one shard per 500 episodes, adjustable with
`--shard-episodes`), capped at the number of Work Items; an explicit count above that cap is rejected.
Quick tests cannot be sharded. Autotune always runs the full, unsharded probe on every instance; sharding
applies only to the scored pass. Each shard records its shard details in `manifest.json` and `result.json`,
including its build: the kit version and the SHA-256 of its benchmark manifest. All shards of one run must
use the same kit version and manifest.

When all shards finish, merge them. The merge checks that items are disjoint and fully covered, that terminal
record IDs are unique, and that each item has the right number of records:

```bash
python -m roboscope_kit.runtime.benchmark.shard_merge \
    --benchmark robotwin --suite demo_clean --benchmark-manifest ... \
    --shard-dir /runs/shard-0 ... --shard-dir /runs/shard-N-1 \
    --output-dir /runs/merged
```

This merge rebuilds the workload from the benchmark manifest, so it needs the same manifest and benchmark
runtime assets (including RoboTwin's seed cache). MetaWorld runs and merges must use the same
`--evaluation-config` file.

The merged `result.json` contains only counts (successes, episodes, steps) plus each shard's provenance and
`result.json` SHA-256. Rates such as `steps_per_s` or `elapsed_s` are never combined across instances; report
cross-instance speedup against the whole job's wall-clock time.

Where neither the benchmark runtime nor the record trees are available, merge from the shard `result.json`
files alone:

```python
from roboscope_kit.runtime.benchmark.shard_reduce import merge_shard_results

merged = merge_shard_results([json.loads(text) for text in shard_result_texts])
```

It requires shard indexes exactly `0..N-1`, identical benchmark metadata (seeds, goals, evaluation
configuration) and package build across shards, and record counts that add up to the workload size, and it
raises `ValueError` otherwise. CALVIN and MetaWorld shards report raw partial counts, because their official
metrics are undefined on a subset; the merge derives the metrics only after confirming full coverage. This
reduction trusts the reported counts and cannot prove that each record is unique; use the record-tree merge
when you need that guarantee.

## Policy HIL model components

The package provides the model side of `policy_hil_continuous`: checkpoint discovery, immutable
verification, Pi0.5 loading, processor construction, and the pinned LeRobot RTC configuration. It is not a
robot-control service and never owns CAN, cameras, recording, progress, or uploads.

A separately packaged robot-side HIL service imports `load_pi05_runtime` and keeps the official
`RTCInferenceEngine`, the model, processor pipelines, robot arms, and cameras in one Python process; there is
no action-chunk wire protocol. Checkpoints and the model resources they reference must live under
explicitly allowed roots. The caller gets a `VerifiedCheckpoint` from `CheckpointCatalog`, records its
hashes in the collection session, and passes it to the component factory together with an explicit local
tokenizer path, because deployed checkpoints may contain paths from the training host. It requires the
pinned LeRobot 0.6.0 used by the validated Pi0.5 RTC deployment.

## Python support

Supported versions follow the [CPython release lifecycle](https://devguide.python.org/versions/): every
released version in its bugfix or security phase. Today that is Python 3.11–3.15. Python 3.8–3.10 are not
supported, and loading the package source through `PYTHONPATH` does not make them supported.

Model frameworks and simulators keep their own pinned interpreters and dependencies; installing
`roboscope-kit` does not install or upgrade them. Simulator environments still pinned to Python 3.8
(CALVIN, RoboTwin) or 3.10 (LIBERO) must be upgraded and validated separately before they can use this
package.

The supported list is declared in `pyproject.toml` under `tool.roboscope.python-support`. On every change,
CI builds one wheel and sdist, installs both on each listed version, imports every module without loading
simulators or models, and runs the regression tests. It also checks the Python 3.11 standard-library
baseline and fails when the list drifts from the official lifecycle.

## Versioning and releases

The names in `roboscope_kit.__all__` are the public API, and so are the command-line arguments this
document describes. Everything else under `roboscope_kit.runtime` is internal; do not depend on its classes.
Installing from PyPI gives you the public `Policy` API, the protocol client, and the benchmark runtime, not
the simulators, datasets, or assets the benchmarks need.

The package follows Semantic Versioning. During the `0.x` Beta series, a minor release may change
`__all__`, the `Policy` lifecycle, or the wire protocol incompatibly; patch releases only fix bugs. Pin
the exact version you validated, and read the
[changelog](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/CHANGELOG.md)
before moving to a new minor version. `1.0.0` will freeze the `Policy` lifecycle and the compatibility
policy.

Read the installed version with `importlib.metadata.version("roboscope-kit")`; `roboscope_kit.__version__`
is not public API.

Releases are published automatically. To release, update the version in `pyproject.toml` and `uv.lock`, add
a changelog section headed `## <version> - <YYYY-MM-DD>` that lists every user-visible API,
`Policy`-lifecycle, and wire-protocol change, and merge to `main`. GitHub Actions then builds the wheel and
sdist, tests them on every supported Python version, uploads them with the `PYPI_API_TOKEN` repository
secret, and checks the published hashes and a clean install. A version already on PyPI is not uploaded
again. To retry a failed upload, run the `SDK compatibility` workflow manually on `main`.

`CHANGELOG.md` ships in the sdist but not in the wheel.

Earlier source releases had a `libero` extra that only installed `imageio[ffmpeg]` and never a runnable
LIBERO simulator, so it was removed. If you use `imageio` directly, declare `imageio[ffmpeg]>=2.34,<3.0` in
your own dependencies.

## License

Copyright 2026 RoboScope Team. Licensed under the Apache License, Version 2.0. See `LICENSE`.
