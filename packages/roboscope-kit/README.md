# RoboScope Kit

`roboscope-kit` is the model-integration boundary for RoboScope's official simulation
benchmarks. Users implement one `Policy` lifecycle and do not install simulator packages or configure
benchmark scheduling and autotune parameters.

Install the SDK from PyPI:

```bash
python -m pip install roboscope-kit
```

## Python compatibility

Formal CPython support follows the upstream
[release lifecycle](https://devguide.python.org/versions/): released versions in
the bugfix or security phase are maintained. The current matrix is Python
3.11–3.15, with Python 3.11 as the minimum supported version. Python 3.8–3.10
are unsupported for installation and runtime use.

Both wheels and source distributions require Python 3.11 or newer. The shared
SDK source and standard-library baseline is Python 3.11. NumPy is resolved to a
version compatible with the selected interpreter. Optional model frameworks
and simulators keep their own separately pinned interpreter/dependency
requirements; installing the SDK does not provision or upgrade them. Pinned
CALVIN/RoboTwin Python 3.8 and LIBERO Python 3.10 environments are outside this
SDK release's support range and need a separately validated environment upgrade
before using it. Loading SDK source through PYTHONPATH does not make those
interpreters supported.

The maintained matrix is declared in
`pyproject.toml` → `tool.roboscope.python-support`. GitHub Actions CI builds
one distribution pair, then on every declared interpreter build/install the
sdist, install the wheel, import all SDK modules without loading simulators or
models, and run the existing SDK/async-inference regressions. A Python 3.11
standard-library compatibility scan and Ruff's Python 3.11 baseline also gate
changes. The lifecycle check reads the official Python release-cycle API and
fails when the maintained matrix drifts, requiring a reviewed update to the
metadata, dependency compatibility and CI before claiming a newly released
version or retiring an upstream-EOL version.

## Policy integration

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


run(UserPolicy())
```

RoboScope launches this entrypoint once per selected inference replica, passes the staged checkpoint and
benchmark contract to `Policy.load()`, and calls it through the local SDK protocol. The user process does
not start a simulator or select evaluator concurrency.

The SDK also contains RoboScope's platform benchmark runtime, durable work queue, process lifecycle,
resource measurement, and automatic K/g selection under `roboscope_kit.runtime`. Those modules
are platform internals and are not exported from the user-facing package API. Installing the PyPI package
provides the public Policy API and protocol client; it does not install simulators, benchmark datasets,
model frameworks, or the platform image dependencies needed to execute those internal runtime modules.

Earlier source releases exposed a `libero` extra, but it installed only `imageio[ffmpeg]`; it never
installed a runnable LIBERO simulator or dataset. The extra has therefore been removed. Existing
environments keep their already-installed packages, but consumers that use `imageio` directly must now
declare `imageio[ffmpeg]>=2.34,<3.0` in their own image or application dependencies. This is only a video
dependency migration and is not a supported way to provision any Benchmark runtime.

The platform runtime uses three core scheduling concepts:

- **Server**: an inference process that loads one user `Policy` replica and serves action requests over a
  local socket. A Server does not run a simulator. Multiple Workers may share one Server, which can batch
  their concurrent requests. Autotune calls the selected Server replica count `g`.
- **Worker**: an evaluator process that owns a benchmark simulator lane. It claims Work Items, runs their
  environments and episodes, sends observations to a Server, applies the returned actions, and publishes
  results. A Worker does not load a separate copy of the user model. Autotune calls the selected Worker
  concurrency `K`.
- **Work Item**: the durable, atomically claimed unit of benchmark work. Its exact size is adapter-specific;
  for example, a RoboTwin Work Item is one manifest-sized episode window from a task. Windows preserve the
  task's canonical seed-cache indices while allowing idle Workers to share long-horizon tasks. A failed
  Worker returns its claimed Work Item to the pending queue, while a completed Work Item is recorded
  idempotently so it cannot be scored twice.

RoboTwin and RoboCasa multi-episode work items publish each completed episode as an observable partial
result. Product progress therefore advances while the item is still running, while official aggregation
and durable completion continue to use only the item's final result. A non-OOM RoboTwin native evaluator
failure with a positive exit code is recorded as a terminal failed item (episodes completed before the
crash are kept and the remainder of that window counted as failures), so the worker continues with the next item instead
of aborting the evaluation pass; OOM and signal-terminated exits still restart the worker and replay its
claimed item, with one retry allowed per runtime pass. Observable progress is recomputed from the durable
queue snapshot and published through the atomic `testlog.txt` status file.

Runtime errors are collected in the append-only `/root/testresult/errors.jsonl` journal. It includes
recoverable Worker failures as well as terminal checkpoint, Policy Server, simulator, configuration, and
other evaluation errors. A failed RoboTwin window keeps the native log excerpt in its worker and native
logs and counts that window's missing episodes as failures in the official aggregate, so Quick Test users see the
simulator or Policy failure instead of only a Worker exit code. Component logs remain available for
detailed diagnosis.

`PolicyCapabilities` declares the supported benchmarks, maximum inference batch size, safe
action-chunk sizes, and optional per-Benchmark action representations:

```python
PolicyCapabilities(
    supported_benchmarks=(BenchmarkName.ROBOTWIN,),
    required_features=("images.head", "state.joints"),
    supported_action_chunk_sizes=(37,),
    action_representations={BenchmarkName.ROBOTWIN: "robotwin.ee-compact-quat-v1"},
    observation_strides={BenchmarkName.ROBOTWIN: 1},
)
```

The runtime validates each declared representation against the Benchmark specification
and requires every loaded replica to resolve the same action contract before
evaluation starts. Existing Policies that omit `action_representations` resolve to
the Benchmark default. The resolved value and its source are recorded in inference metadata.
RoboTwin also accepts a positive per-Policy observation stride. Its safe default
is 1, and every new inference chunk refreshes the current simulator observation.
The evaluator renders one observation per refresh. Randomized lighting, the viewer,
video logging and point-cloud observations keep the per-step cadence regardless of
the declared stride.

Evaluator concurrency and replica count are platform-owned autotune parameters rather
than user-declared capabilities. SDK-native scored evaluations measure real Policy-backed episodes
across K/g candidates. Every candidate runs the same deterministic, representative episode set. The
minimum safe K establishes a success-rate baseline, and later candidates must retain at least 85% of that
baseline while improving throughput and keeping resources safe. Candidates that regress beyond this 15%
tolerance, lose Workers, report a Worker failure or OOM, or leave probe episodes incomplete are rejected.
The platform stops a faulty probe immediately, limits each candidate to 15 minutes and the complete tune to
30 minutes, then continues with the last healthy K/g pair. `/root/testresult/autotune/selection.json` is
updated atomically when every candidate starts, advances, or stops, so running, accepted, rejected, and timed-out trials
remain visible to platform log synchronization without contributing episodes to the formal score.

## CUDA MPS (opt-in)

`--cuda-mps` runs the Policy replicas under NVIDIA CUDA MPS, so kernels from several replicas on one
GPU run concurrently instead of time-slicing. It is off by default; `run-benchmark.sh` accepts the same
flag. When it is set, the runtime:

- starts a private `nvidia-cuda-mps-control` daemon for the visible GPUs, with 100% active-thread access,
  and refuses to start if another MPS daemon or server is running;
- attaches every Policy replica to that daemon and fails the run if a loaded replica is not an MPS client;
  simulator Workers are not attached;
- runs autotune probes and the formal pass in the same mode, so K/g is selected for MPS;
- terminates a replica's MPS clients through the daemon before stopping or rebuilding that replica, then
  quits the daemon when the run ends;
- writes daemon logs to `<output-dir>/cuda-mps/` and records the session under `cuda_mps` in `result.json`
  (`{"enabled": false}` when off).

The GPU must be in the Default or Exclusive-Process compute mode, and the host must provide the
`nvidia-cuda-mps-control` binary. Replicas select their GPU by ordinal within the daemon's devices, so a
Policy must not override `CUDA_VISIBLE_DEVICES`.

## Cross-instance episode sharding (Beta)

Cross-instance sharding is currently **Beta**, released in SDK `0.7.0b48` (then published as
`roboscope-inference-sdk`; `0.8.0b1` is the first release published as `roboscope-kit`).
To install this prerelease explicitly:

```bash
python -m pip install 'roboscope-kit>=0.8.0b1'
```

On the platform, first enable Beta features in account settings, then enable sharding
for the individual evaluation and choose its instance count. Ordinary evaluations remain
unsharded. Platform availability also requires a released, verified image and launcher;
publishing the SDK alone does not enable this feature in production.

The benchmark CLI can split one evaluation deterministically across independent GPU instances,
removing the single-instance GPU-count ceiling. Every instance generates the identical full
work-item list and keeps only the items whose list position satisfies `position % N == shard_index`;
the partition therefore needs no coordination, and RoboTwin seed assignment (absolute episode
index into the validated seed-cache generation) stays identical to a single-instance run.

Run one instance per shard index:

```
python -m roboscope_kit.runtime.benchmark \
    --benchmark robotwin --suite demo_clean --checkpoint-source ... \
    --shard-index I --shard-count N --output-dir /runs/shard-I
```

Omit `--shard-count` to derive it from the workload size (one shard per 500 episodes,
configurable with `--shard-episodes`), capped at the number of work items. An explicit count
above that limit is rejected on every shard. Quick tests cannot be sharded. Autotune probes always run the unsharded probe
workload on every instance; sharding applies to the formal pass only, and each shard run
records its shard metadata in `manifest.json` and `result.json`.

After every shard completes, merge and validate the partition (disjoint items, full
coverage, unique terminal record ids, per-item record counts) with:

```
python -m roboscope_kit.runtime.benchmark.shard_merge \
    --benchmark robotwin --suite demo_clean \
    --shard-dir /runs/shard-0 ... --shard-dir /runs/shard-N-1 \
    --output-dir /runs/merged
```

Record-tree merging reconstructs the workload using the supplied image manifest and requires
the same benchmark runtime assets (including RoboTwin's validated seed cache). MetaWorld runs
and record-tree merges must both supply the same resolved `--evaluation-config` file.

Official images from SDK 0.7.0b51 freeze RoboTwin's 50 × 100 inputs per suite in
the image under `robotwin2-platform-native-step-lim-v2`. The clean entries are
preserved; the randomized pool is newly generated. Report randomized v1 and v2
scores separately. Input provenance and exact digests are documented in
[the fixed-input protocol](../../benchmarks/starvla_platform_runtime/robotwin/SEED_PROTOCOL.md).

The merged `result.json` contains count metrics only (successes, episodes, steps) plus
per-shard provenance including each shard `result.json` SHA-256. Rate metrics such as
`steps_per_s` or `elapsed_s` are never aggregated across instances; each shard keeps its
own values, and cross-instance speedup is reported against the whole job's wall-clock time.

To merge from the shard `result.json` documents alone (for example on a control plane that has
neither the benchmark runtime nor the record trees), use `merge_shard_results`:

```python
from roboscope_kit.runtime.benchmark.shard_reduce import merge_shard_results

merged = merge_shard_results([json.loads(text) for text in shard_result_texts])
```

It checks that the shard indexes are exactly `0..N-1`, that all shards describe the same
complete benchmark metadata (including seeds, goals and evaluation configuration) and SDK build,
and that their individual and combined record counts add up to the workload size every
shard recorded, then returns the merged count metrics and official result. It raises
`ValueError` on any inconsistency. CALVIN and MetaWorld shards report raw partials instead of
final metrics, because their official metrics are undefined on a subset of the workload; the
merge asserts full sequence/task coverage before deriving them. Result-only reduction checks
reported counts and metadata; it cannot prove per-record uniqueness or ownership without the
completed-record trees. Use record-tree merging when that stronger validation is required.

## API stability and versioning

The names exported by `roboscope_kit.__all__` are the public SDK API. Modules below
`roboscope_kit.runtime` are platform implementation details unless this document explicitly
states otherwise; applications should not depend on their internal classes or command-line arguments.

The package follows Semantic Versioning. During the current `0.x` Beta series, a minor release may make an
incompatible change to `__all__`, the public Policy lifecycle, or the wire protocol, while patch releases
remain compatible bug fixes. Every evaluation image pins one exact SDK version. User images should also
pin the version they validated, and review release notes before upgrading to another minor version. The
first `1.0.0` release will freeze the public Policy lifecycle and compatibility policy for stable use.

The installed version comes from distribution metadata and can be read with
`importlib.metadata.version("roboscope-kit")`; `roboscope_kit.__version__` is not part
of the public API. The release owner updates the version in `pyproject.toml` and `uv.lock`, moves the
`CHANGELOG.md` Unreleased entries into a dated release section, and records every user-visible API,
Policy-lifecycle, and wire-protocol change in the same release merge request.

Releases are published automatically. When a commit on `main` carries a version that is not yet on PyPI,
GitHub Actions builds the wheel and sdist once, runs the full compatibility matrix on them, uploads them
with the `PYPI_API_TOKEN` repository secret, and then checks the published hashes and a clean install.
Commits whose version is already on PyPI skip the upload. To retry a failed release, run the
`SDK compatibility` workflow manually on `main`.

`CHANGELOG.md` is included in the source distribution, but is intentionally not copied into the wheel's
runtime package. Wheel users can read the same release notes in the repository or PyPI source archive.

## Episode-scoped state

The runtime calls `reset(EpisodeContext)` before an episode, sends its `episode_id` on each observation,
and calls `end_episode(episode_id)` when the episode finishes. The SDK forwards this lifecycle but does not
store or synchronize mutable Policy state.

Stateful Policies should isolate mutable state by `episode_id` and release it from `end_episode()`.
Stateful and stateless Policy examples are provided in the RoboScope user manual.

## Model-owned input preprocessing

Each Benchmark sends the canonical observation defined by its Benchmark contract without model-specific
resize, normalization, or channel/layout conversion. The user Policy performs those transformations before
calling its model. Model input dimensions therefore do not belong in the public Benchmark or Policy API.

## Current benchmark status

The repository contains the following platform execution chains. Their manifests describe dependencies
that RoboScope's Benchmark images or shared storage must provide; they are not dependencies of the PyPI
distribution. A chain being present here does not guarantee that every deployment has provisioned its
external artifacts.

| Benchmark | Platform execution chain | Image/shared-storage dependencies |
| --- | --- | --- |
| LIBERO | SDK-native adapter and persistent LIBERO worker | Pinned LIBERO Python environment, simulator source, dataset and initial states |
| LIBERO+ | SDK-native adapter that reuses the LIBERO worker | Pinned LIBERO+ Python environment, simulator archive, dataset and initial states |
| RoboTwin | SDK-native adapter and worker that invoke the pinned RoboTwin launcher | RoboTwin environment and source, deploy policy, checkpoint assets and validated seed cache |
| RoboCasa | SDK-native adapter and persistent RoboCasa worker | Pinned RoboCasa environment, simulator/task source and dataset assets |
| CALVIN | SDK-native adapter and persistent CALVIN worker | Pinned CALVIN environment, benchmark/config source, dataset and evaluation-sequence manifest |
| MetaWorld | SDK-native adapter and persistent MetaWorld MT50 worker | Pinned MetaWorld/MuJoCo environment, task assets and the frozen MT50(seed=42) train-goal artifact |

LIBERO and CALVIN convert their native observations to the canonical SDK feature names and send native
7-dimensional actions back to their simulators. CALVIN's task-D-to-task-D path implements the official
1,000 long-horizon sequences with static/gripper images, proprioception and language. The suite list and
runtime paths for every chain remain owned by its installed Benchmark image manifest.
- Custom-image users provide their own SDK `Policy` and saved image. Evaluation uses platform Benchmark
  environments and evaluators from `/autodl-fs`.

The complete LIBERO entrypoint and scoring contract is documented in
`LIBERO_EVALUATION_PROTOCOL.md` in the RoboScope Kit repository.
The LIBERO document defines the `/root/startbenchmark.sh` arguments, checkpoint staging, canonical
observation/action mapping, episode initialization and horizons, success detection, output files, error
codes, and required validation sequence.
The SDK-native CALVIN manifest, observation/action mapping, official long-horizon evaluator, aggregation,
and validation procedure are documented in
`CALVIN_EVALUATION_PROTOCOL.md` in the RoboScope Kit repository.
The SDK-native MetaWorld MT50 manifest, camera configuration, official training-goal evaluation,
four-difficulty macro aggregation, and validation procedure are documented in
`METAWORLD_EVALUATION_PROTOCOL.md` in the RoboScope Kit repository.

The SDK integration, autotune, and benchmark-runtime throughput results are maintained in
`docs/benchmark-results.md` in the RoboScope Kit repository. The report keeps LIBERO and RoboTwin results separate and includes the StarVLA official
four-GPU RoboTwin launcher baseline used for the runtime comparison.

## Policy HIL model components

The SDK supplies checkpoint discovery, immutable verification, Pi0.5 loading,
processor construction, and the pinned LeRobot RTC configuration used by
`policy_hil_continuous`. It does not provide a robot-control service and never
owns CAN, cameras, recording, progress, or uploads.

A separately packaged robot-side HIL service imports `load_pi05_runtime` and
keeps the official `RTCInferenceEngine`, model, processor pipelines, robot arms,
and cameras in one Python process. There is no action-chunk wire protocol or
queue adapter.

Local checkpoints and referenced model resources must remain inside explicitly
allowlisted roots. The caller obtains a `VerifiedCheckpoint` from
`CheckpointCatalog`, freezes its hashes into the collection session, and passes
it to the component factory. The factory also requires an explicit local
tokenizer path because deployed checkpoints can contain training-host paths that
are not valid on the collection machine.

The runtime requires the pinned LeRobot 0.6.0 installation used by the validated
native Pi0.5 RTC deployment. The robot-side HIL contract belongs to that
service; this package exposes model components rather than a socket server.

## Policy process contract

Custom images contain the user's SDK `Policy`, dependencies and `/workspace/roboscope-policy.json`.
The LIBERO platform entrypoint loads the Policy process contract from the fixed image path
`/workspace/roboscope-policy.json`:

```json
{
  "schema_version": 2,
  "command": ["/opt/user/bin/python", "/workspace/user_policy.py"],
  "cwd": "/workspace"
}
```

RoboScope selects the Policy runtime and validates the file independently during quick test and scored
evaluation. End users do not configure the path, K/g, or benchmark runtime flags.

## License

Copyright 2026 RoboScope Team. Licensed under the Apache License, Version 2.0. See `LICENSE`.
