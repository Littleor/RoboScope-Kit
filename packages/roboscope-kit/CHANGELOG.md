# Changelog

## 0.8.0b15 - 2026-11-04

- RoboDojo builds no longer refuse accounts with `$HOME/miniconda3`. `build-runtime.sh` removes the upstream
  installer's activation of that path, and ignores `PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV`.

## 0.8.0b14 - 2026-11-04

- Wait up to 15 minutes for the new version on the PyPI index in the post-release install check. The previous
  3-minute wait was shorter than the index's 10-minute CDN cache, so 0.8.0b13 failed the check after publishing
  correctly. The package is unchanged.

## 0.8.0b13 - 2026-11-04

- Add `examples/xpolicylab/`, a Policy that serves an XPolicyLab policy from the pinned RoboDojo checkout on
  the `robodojo` benchmark, following XPolicyLab's reference episode loop through per-step observation
  updates. `Simate_beta` is the validated configuration. The package runtime is unchanged.

## 0.8.0b12 - 2026-11-04

- Remove the unused `calculate_k_limits`, `KLimits`, `calculate_g_limit`, and `build_adaptive_k_candidates`
  from `roboscope_kit.runtime.autotune`. They were internal and no runtime path called them.

## 0.8.0b11 - 2026-11-03

- Add RoboDojo simulation support through the standard Policy and benchmark runtime. The pinned manifest
  covers 42 base tasks and 12 randomized variants with their upstream episode limits. Policies may declare
  per-step observation updates; asynchronous observation delivery and several upstream runtime candidates
  remain opt-in. The runtime patches are adapted from the RoboDojo work by @QLL-CCDD.

## 0.8.0b10 - 2026-11-01

- Use the diagonal search on every benchmark. CALVIN, LIBERO-Plus, MetaWorld, and RoboCasa probes now grow
  with K, as LIBERO and RoboTwin probes do, so every probed worker gets `probe_items_per_worker` work items
  and the steady-state estimator has a window. CALVIN takes a longer prefix of the evaluation sequences
  (at most 1,000); LIBERO-Plus, MetaWorld, and RoboCasa take a longer prefix of one nested task order
  (`nested_task_prefix`), so each probe repeats the episodes of every smaller probe. With three items per
  worker, RoboCasa's 24 tasks cover up to K=8 and MetaWorld's 50 tasks up to K=16. The four manifests no
  longer pin `autotune_strategy: two-phase`. With `two-phase`, the probes keep their fixed size.

## 0.8.0b9 - 2026-11-01

- Make `diagonal` the default autotune strategy. On LIBERO-Long with one RTX 4090 it selected K12/g3 in
  three of three launches, against K4/g1 or K9/g1 for `two-phase`, and evaluated 1.81x faster. The CALVIN,
  LIBERO-Plus, MetaWorld, and RoboCasa manifests pin `autotune_strategy: two-phase`, because their probes
  are not sized by K and so give the steady-state estimator no window at higher K. A manifest without
  `autotune_strategy` now gets `diagonal`, the `steady` throughput estimator, and
  `probe_items_per_worker: 3`.

## 0.8.0b8 - 2026-10-31

- Run the kit standalone, with no hosted platform:
  - The CLI takes every path as an argument: `--benchmark-manifest` (required, also for `shard_merge`),
    `--policy-config` for the Policy process contract, and `--checkpoint-dir` for an `oss://` checkpoint.
    `load_policy_runtime_config(path)` takes the path; `POLICY_CONFIG_PATH` is removed.
  - The benchmark manifest is `benchmark-manifest.json`, schema 3 (`BenchmarkManifest`,
    `load_benchmark_manifest`, module `runtime.benchmark.manifest`). It drops `image_name`,
    `evaluation_kit_commit` and `sdk_version`, and relative `runtime` paths resolve against the manifest's
    directory. RoboTwin runs its launcher with the kit's own interpreter, so its `runtime.python` is gone.
  - Each benchmark lives in `benchmarks/<benchmark>/` with its manifest and a `build-runtime.sh <install-dir>`
    that builds the pinned simulator environment and copies the manifest there. LIBERO gets its first build
    script; RoboTwin's launcher and seed tools move from `benchmarks/robotwin_official/`.
  - A shard records its build as the kit version and the manifest's SHA-256, and the merge requires both
    to match.
  - Simulator workers import the kit from a directory that holds only `roboscope_kit`, so a pip-installed
    kit no longer exposes its `site-packages` to the simulator's interpreter.
  - Removed: the `testlog.txt` progress contract and `--legacy-contract-root`, the shared-disk Control
    Runtime installers and launchers, and the old LIBERO entrypoint `benchmarks/libero_official/`.

## 0.8.0b7 - 2026-10-31

- Evaluate at a fixed K/g with `--concurrency K --replicas G`, which skips autotune probing and records
  `mode: fixed` in `result.json`, so one configuration can be measured or reproduced through the public
  CLI. Both flags are required together and cannot be combined with `--quick-test` or
  `--autotune-strategy`.

## 0.8.0b6 - 2026-10-31

- Add the opt-in `diagonal` autotune strategy (`autotune_strategy: diagonal`). It ramps workers on one
  replica per GPU until doubling K gains less than `autotune_worker_knee_gain` (25%); a step the worker
  cap shortens must gain the same share of linear scaling. It then adds one replica per GPU at a time with
  K scaled by the measured workers per replica, keeping a step while it gains at least
  `autotune_replica_minimum_gain` (3%). The worker cap comes from the marginal CPU cost of one worker,
  fit from two probes, so policy-server and controller CPU is not charged to workers; before two probes
  exist it uses the manifest's per-worker CPU estimate. A step whose busiest GPU would run out of memory,
  extrapolated from the replica footprint and the marginal memory per worker, is skipped. Probes the
  strategy measures but does not keep are marked `rejected` in `selection.json`, the selection reports
  the selected probe's bound hint, and replica-phase probes are listed in `g_trials`. `two-phase` stays
  the default for every benchmark.
- Choose the autotune strategy per evaluation with `--autotune-strategy`, which overrides the manifest's
  `autotune_strategy` without changing the manifest file. An unknown name fails before the evaluation
  starts.
- Put autotune strategies behind one interface. `register_strategy(name)` adds a K/g search that a
  manifest selects with `autotune_strategy: <name>`; every strategy measures through the shared
  controller, which owns probe execution, resource and correctness checks, the time budget and
  `selection.json`. `selection.json` records the strategy and the throughput estimator in `limits`, and both
  strategies write the same `bounds`, which now include `max_workers_hard`. `select_k` and the diagonal
  strategy share one set of CPU, GPU-utilization and throttle limits.
- Measure probes at steady state. `steady_steps_per_s` (with its `steady_window`) is the throughput of
  K concurrently busy workers: each worker's first item, which carries environment creation and Policy
  warm-up, is dropped, and so are the items that finish after the first worker runs out of work. Select
  it with `autotune_throughput_estimator: steady`; it is the default for the diagonal strategy, and a
  probe without a steady-state window is rejected as `no_steady_window`. The default `elapsed`
  estimator of `two-phase` is unchanged.
- Size LIBERO and RoboTwin probes by K with `probe_items_per_worker`, so every probed worker gets that
  many work items. It defaults to 3 (one warm-up and two measured items) with the steady estimator and
  to 0, the fixed probe size, otherwise. LIBERO keeps its task set and adds episodes per task, counting
  `chunk_size` episodes per item, capped at the 50 initial states of a task. RoboTwin first adds tasks
  from one nested order, then whole work items per task, capped at the episodes of the formal
  generation. Either way every probe repeats the episodes of each smaller probe.
- Compare a probe's success rate with the baseline on the episodes both ran (`record_success` in
  `result.json`, `matched_episodes` in the probe health). Probes with identical workloads, the default,
  compare exactly as before.
- Record per-item completion times in `result.json` (`item_timeline`): the time the controller first sees
  each work item completed. Partial results, which RoboTwin and RoboCasa publish per episode, do not count.
  The controller reads completed items from the same queue scan it uses for progress.

## 0.8.0b5 - 2026-10-31

- Add one StarVLA demo per benchmark under `examples/starvla/<benchmark>/`: LIBERO, LIBERO-Plus, RoboTwin,
  RoboCasa, CALVIN, and MetaWorld. Each `demo.yaml` pins a published checkpoint, the StarVLA revision, and
  the inference configuration; `prepare_snapshot.py` builds the run snapshot from it.
- Add `starvla_libero_policy.py`, the LIBERO Policy behind the technical report's LIBERO-Long results. It
  follows the StarVLA `eaa51c4` LIBERO client.

## 0.8.0b4 - 2026-10-31

- Retire the custom-image and image-baking paths. Move the X-VLA example from
  `examples/custom_image_validation/xvla/` to `examples/xvla/` and drop its baking script. Remove the LIBERO
  base-image bundle scripts (`build_official_image_bundle.sh`, `install_official_image_bundle.sh`,
  `prepare_official_image_for_bake.sh`, `verify_official_image.sh`) and the related documentation.

## 0.8.0b3 - 2026-10-31

- Add the StarVLA Policy as an example under `examples/starvla/`, restored from the removed
  `official_images/starvla/` source. It is the Policy behind the StarVLA measurements in the RoboScope Kit
  technical report.

## 0.8.0b2 - 2026-10-30

- Remove the official StarVLA image source (`official_images/starvla/`) and its documentation. Custom
  images are the only documented evaluation path. The X-VLA custom-image demo now installs its Control
  Runtime from the official base-image bundle (`benchmarks/libero_official/build_official_image_bundle.sh`).

## 0.8.0b1 (Beta, published 2026-10-30)

- Rename the SDK to RoboScope Kit. Install it with `pip install roboscope-kit` (previously
  `roboscope-inference-sdk`) and import it as `roboscope_kit` (previously `roboscope_inference_sdk`).
  `importlib.metadata.version()` now takes `"roboscope-kit"`. No compatibility alias is provided.
- Move the package source from `packages/roboscope-inference-sdk/` to `packages/roboscope-kit/`; image
  Control Runtimes now live under `/opt/roboscope/control-runtime/packages/roboscope-kit/`.

## 0.7.0b77

- Add an opt-in CUDA MPS mode for Policy replicas: `--cuda-mps` on the
  benchmark CLI and `run-benchmark.sh`, off by default. The runtime owns a
  private MPS control daemon for the run, attaches every replica to it,
  fails if a replica does not attach, terminates a replica's MPS clients
  before signalling it, and records the session in `result.json`. Autotune
  probes run under the same mode as the formal pass.

## 0.7.0b76

- Revert the MetaWorld runtime, autotune, native-reference and MPS changes
  because they regressed the autotune algorithm. Restore the b59 runtime
  and autotune behavior.

## 0.7.0b59

- RoboTwin renders one observation per refresh. The evaluator loop now asks the
  Policy client whether an observation is due, so steps at chunk or stride
  boundaries no longer render twice and stride 1 matches the official cadence.
- Keep the per-step observation cadence under randomized lighting, the viewer,
  video logging and point-cloud observations, whose output or RNG stream depends
  on every render.
- The StarVLA official Policy declares its action horizon as its observation
  stride, replacing the external stride-16 launcher wrapper.

## 0.7.0b58

- Move PyPI upload, artifact-hash verification and clean-install verification to
  GitHub Actions after the full compatibility gate.
- Require Python 3.11 or newer for installation and runtime use; remove the
  Python 3.8–3.10 compatibility tier and its CI/dependency branches.
- Define maintained CPython support by the upstream lifecycle, currently
  Python 3.11–3.15, and align package metadata and Ruff with that baseline.
- Use Hatchling for wheel and source-distribution builds with SPDX metadata.
- Select the locked Control Runtime NumPy wheel by its CPython/Linux tags so
  simplified lockfile markers do not break official image bundle builds.
- Gate changes on installed-artifact imports, the existing regression suites,
  Python 3.11 API checks and the declared interpreter matrix in CI.
- Retain strict zip length validation; remove the Python 3.8-only API workarounds
  from this candidate. Older pinned simulator environments require separate
  upgrade and image validation before adopting this SDK.

## 0.7.0b50 (Beta)

- Add reproducible builders for CALVIN, RoboCasa, RoboTwin and LIBERO+ runtimes.
- Lock source revisions, asset checksums and the restored CALVIN dependency set.
- Include RoboCasa kitchen/GR1 assets and RoboTwin embodiment path preparation
  in their builders; use CALVIN's validation long-finger Panda configuration.
- Align RoboTwin Quick Test preparation with its single-task workload.
- Include all RoboTwin preparation scripts in the shared Control Runtime so its
  source checksum matches the committed checkout.
- Reuse the commit-pinned image bundle for offline RoboTwin preparation runtime
  installation instead of downloading CPython and numpy again.

## 0.7.0b49 (Beta)

- Restore the validation long-finger Panda geometry in CALVIN runtime preparation.

## 0.7.0b48 (Beta)

- Cross-instance sharding is an opt-in Beta. SDK publication does not enable platform sharding
  or promote any evaluation image to the official catalog.

- Bound automatic shard counts by work-item capacity and reject invalid partitions before autotuning.
- Validate shard protocol identity, per-item ownership and per-shard accounting before merging.
- Accept MetaWorld evaluation configuration in the record-tree merge CLI.

## 0.7.0b47

- A shard run records the image's SDK version and evaluation kit commit (from the image manifest)
  in its `shard.build`, and `merge_shard_results` refuses shards that ran different builds or do
  not record one, naming each shard's build. The merged result carries the common build.
- New `roboscope_inference_sdk.runtime.benchmark.suite_sizes.suite_total_records(benchmark, suite)`:
  how many records one formal run of a suite produces, for callers that must split a run before any
  instance exists (the platform sizes a sharded run from it instead of keeping its own copy).
  Covers LIBERO, RoboTwin, CALVIN, RoboCasa and MetaWorld MT50 (50 tasks x 50 episodes).
  `test_benchmark_suite_sizes` fails when the table drifts from the image manifests or the runtimes'
  task lists, or when an image manifest is neither sized nor declared unsized. LIBERO+ has no fixed
  size and returns `None`.

## 0.7.0b46

- New `roboscope_inference_sdk.runtime.benchmark.shard_reduce.merge_shard_results`: merges the
  `result.json` of every shard into one result using only those documents, with no simulator,
  benchmark factory or record tree on the merging machine. Coverage is established by accounting
  (shard indexes exactly `0..N-1`, one benchmark/suite/protocol, record counts summing to the
  `full_total_records` each shard recorded). `shard_merge` remains the record-level validator.
- CALVIN and MetaWorld now support sharding. Their `aggregate()` demands the whole workload and
  raised on a shard's subset, so a sharded run failed after the evaluation had finished. Sharded
  runs now call an optional adapter `aggregate_partial()`, which reports raw per-sequence lengths
  (CALVIN) or per-task counts (MetaWorld) and withholds metrics that are undefined on a subset
  (`avg_seq_len`, `success_at_5`, the MetaWorld difficulty macro `score`). The official result is
  derived after merging, where full coverage of every sequence or task is asserted. Single-instance
  `aggregate()` still requires complete coverage.

## 0.7.0b45

- Cross-instance episode sharding for the benchmark runtime: `--shard-index` with
  `--shard-count` (or workload-derived, one shard per `--shard-episodes` episodes)
  deterministically partitions the formal evaluation pass across independent GPU
  instances; autotune probes stay unsharded on every instance.
- New `roboscope_inference_sdk.runtime.benchmark.shard_merge` module: validates the
  shard partition (disjoint items, full coverage, terminal accounting) and merges
  shard results with count metrics only, keeping per-shard rate metrics as provenance.
- The legacy `testlog.txt` contract now counts a shard's own partition instead of the
  full workload. It previously reported the unsharded total, so every sharded run
  showed the whole suite as its denominator and then failed its completeness check
  with `legacy protocol expected N results, received N/shards`.

## 0.7.0b44

- Add MetaWorld MT50 support to the official StarVLA Policy image: the
  benchmark-name image preprocessing dispatch now covers `metaworld`, and the
  official image declares the pinned StarVLA MetaWorld reference observation
  pipeline (corner2, 480 render, 180-degree rotation, 2/3 center crop,
  224x224 output) as its `benchmark_config.metaworld` default.
- Ship the pinned upstream MetaWorld data registry (`metaworld_mt50` mixture
  and `metaworld_robot` DataConfig) into the official image so StarVLA
  MetaWorld checkpoints resolve their dataset mixture at Policy load; the
  image install and verify scripts cover the installed file.

## 0.7.0b43

- Add the MetaWorld MT50 platform benchmark (`metaworld`, protocol
  `metaworld-v1`, suite `mt50`). The SDK-native adapter reuses the pinned
  upstream `metaworld.evaluation.evaluation` loop over one frozen
  `MT50(seed=42)` training-goal artifact and scores the four-difficulty macro
  average (`difficulty_macro_success_rate`).
- MetaWorld observations come from the resolved evaluation config: one to four
  configurable `uint8` camera features plus fixed `float32[4]` proprioception and
  the verbatim official task instruction.
- Add `--evaluation-config` to the unified benchmark runtime entrypoint. The
  resolved configuration feeds both the evaluator workers and the Policy `load`
  session, so both sides see the same observation features.
- Bump the Policy process contract to `schema_version` 2 with an optional
  `benchmark_config` declaration; readiness/preflight returns the declaration
  without importing the Policy or loading a model.

## 0.7.0b41

- Restore shared-runtime archive timestamps after flushing extracted data on
  AutoDL FS. Keep full tar comparison as the recovery acceptance gate for
  benchmark runtimes and the shared Control Runtime.

## 0.7.0b40

- Shared-runtime rollout tooling: select one benchmark per operation, retain
  the expanded runtime through end-to-end acceptance, and provide explicit
  rollback and finalization. Preserve complete recovery archives and original
  preparation scripts; reclaim the shared Control Runtime as a separate batch.
- The production compaction rollout retains the existing b31 live manifests,
  launchers and image runtime. This release changes operator tooling only.

## 0.7.0b39

- Revert the PI0.5 checkpoint-loading changes: restore the prior PI0.5 checkpoint loading and
  normalizer lookup behavior. Retain the Python 3.10 timestamp import fix.

## 0.7.0b38

- Policy HIL: use `timezone.utc` for checkpoint timestamps so the SDK imports
  successfully on Python 3.10.

## 0.7.0b37

- Policy HIL: resolve normalizer weights from the selected checkpoint's
  declared preprocessing pipeline instead of assuming a fixed step index.

## 0.7.0b36

- Policy HIL: construct PI0.5 on empty parameters and assign safetensors directly
  on the target device, avoiding random initialization and the subsequent full
  checkpoint copy while retaining strict key and materialization checks.

All notable user-visible changes to RoboScope Kit releases are
recorded here. Versions follow Semantic Versioning; the compatibility rules
and release-owner responsibilities for the current Beta series are documented
in `README.md`.

## Changes between 0.7.0b5 and 0.7.0b36

- AutoDL shared runtimes: store the expanded RoboCasa, RoboTwin, and LIBERO+
  runtime roots as verified zstd level-3 tar archives, extract the selected
  runtime onto the instance-local expanded system disk, and keep the shared
  Control Runtime installer workspace local. RoboTwin excludes retired image
  and result trees from its cold-start payload. This releases the shared file
  storage from its 200,000-inode design limit while minimizing transfer time.
- Benchmark docs: remove the retired `docs/robotwin-vectorized-runtime.md`
  archive and the redundant `docs/benchmark-results.html` mirror, and fold
  the archived report's benchmark machine specification, bit-exact fidelity
  verification, champion resource profile, and cache-grow/recycling analysis
  into `docs/benchmark-results.md`.
- Official image bundles now install and checksum the RoboTwin native/cache
  paired verifier alongside the exact seed scanner used by formal generation.
- RoboTwin runtime preparation: reconstruct the protocol-pinned pristine
  checkout from committed Git objects for every run, apply the deterministic
  Python 3.8 annotation patch as a fixed derived commit, attach large simulator
  assets as read-only inputs, and generate each formal seed generation against
  that clean checkout. The launcher now derives its complete evaluator
  instrumentation from the pinned source instead of relying on manual
  shared-disk patches.
- Benchmark autotune: run every concurrency candidate against the same fixed,
  representative workload and reject candidates whose success rate falls more
  than 15% below the minimum safe-concurrency baseline. This keeps stateful
  third-party Policies at the last correctness-preserving K/g pair.
- Official runtime ownership: move the live StarVLA LIBERO runtime under
  `benchmarks/libero_official` and remove the obsolete benchmark-autotune
  examples tree and retired JSON evidence.
- Policy action contracts: every Policy declares one action representation per
  supported Benchmark. The runtime validates the identifier and requires all
  loaded replicas to resolve the same contract before evaluation starts.
- Benchmark action routing: RoboTwin and CALVIN adapters translate the
  negotiated action contract into their official worker execution modes.
- RoboTwin observation cadence: Policies may declare a positive observation
  stride. Every inference boundary refreshes the current observation, and the
  resolved value and source are recorded in execution metadata.
- Benchmark protocol manifests: schema version 2 carries benchmark-owned,
  versioned protocol definitions. RoboTwin pins its runtime, scanner source,
  suite seed selection, episode count, horizon, success rule, and aggregation.
- Immutable RoboTwin seed generations: the SDK can create and validate
  content-addressed cache directories whose manifest covers every task pickle,
  runtime source, scanner source, task configuration, and seed-selection rule.
- Formal RoboTwin runs now select one immutable seed generation by digest,
  validate its full 50-task identity before workers start, and record the
  protocol and generation identities in result metadata.
- RoboTwin cache provenance tooling now ships the exact scanner source in the
  official bundle and provides a paired native/cache episode-state verifier.
- Documentation and examples: classify model-specific custom-image validation
  demos as internal engineering evidence, separate from the official image
  catalog and product support matrix.
- RoboTwin full runs: split each task into manifest-sized episode windows and
  interleave windows across tasks. Records retain their canonical per-task
  episode indices and seed-cache entries, while idle workers can take later
  windows from long-horizon tasks instead of waiting for one worker to finish
  all 100 episodes.
- RoboTwin cached evaluation: count a cached episode whose real scene setup is
  unstable as a zero-step failure and continue to the next canonical cache
  entry. The runtime no longer retries the same randomized seed forever or
  substitutes a different seed to complete the requested episode count.
- RoboTwin persistent sessions: apply worker-static seed-cache and video
  settings before launcher startup. Seed-cache-only planner setup is now
  enabled in session mode, and `--disable-video` no longer starts per-episode
  ffmpeg processes. Each completed window also closes its task-scoped Policy
  socket instead of accumulating idle client threads for the life of the
  launcher.
- Benchmark process lifecycle: make SDK-managed processes own their complete
  process group and keep RoboTwin evaluator sessions inside the worker group,
  so a rejected or timed-out autotune candidate fully releases simulator GPU
  contexts before the next candidate or formal evaluation starts. OOM
  classification now scans the complete diagnostic tail instead of losing an
  earlier CUDA allocation failure behind a final secondary traceback.
- Benchmark autotune: bound every real-episode candidate to 15 minutes and the
  complete tuning phase to 30 minutes, reject OOM/Worker-failure candidates as
  soon as the runtime observes them, and atomically persist accepted, rejected,
  timed-out, and running candidate state so the platform can audit partial
  tuning progress. A rejected or timed-out candidate falls back to the last
  healthy K/g selection instead of failing the formal evaluation.
- RoboTwin worker/launcher: add an opt-in persistent evaluation session
  (`--eval-session`, enabled automatically for full runs with a seed cache)
  that keeps one launcher process alive across task work items. Combined with
  the new shared CuroboPlanner patch, the EE path pays cuRobo planner
  construction once per worker instead of once per task; a crashed task ends
  the session and the next task starts in a fresh process.
- RoboTwin launcher: skip the expert-only batch cuRobo `MotionGen` warmup
  during planner construction when the EE path runs with a seed cache. The
  batch instance is only reachable through the expert `play_once` grasp
  search, which the cache replaces, so its warmup is pure per-task planner
  construction overhead; `plan_single` EE planning semantics are unchanged.
- RoboTwin launcher: add the explicit image-owned `ROBOTWIN_ACTION_TYPE=ee`
  path for compact dual-arm end-effector actions while keeping qpos as the
  platform default. EE execution uses the official `take_action(...,
  action_type="ee")` and cuRobo planner even with a seed cache.
- CALVIN worker: add the image-owned `CALVIN_ACTION_TYPE=absolute` opt-in that
  reconstructs an absolute `(pos, quat_xyzw, grip)` tuple from a compact 7-dim
  action chunk, validating the quaternion xyz norm and rebuilding the
  non-negative `w`. The platform default stays the native relative 7-dim
  action contract.
- LIBERO runtimes: prepare LIBERO and LIBERO+ with run-local configuration and
  verify package, SDK worker, and benchmark asset paths in the actual simulator
  Python before evaluation. Shared simulator environments remain immutable and
  no longer depend on a baked LIBERO+ configuration path.
- Policy server: default `TORCHINDUCTOR_CUDAGRAPHS=0` on every policy replica
  process. Several simulator workers share one torch.compile policy
  replica; concurrent first-inferences of different input shapes each trigger a
  new CUDA Graph capture, and capture is device-level exclusive, so the workers
  collide with `beginAllocateToPool: already recording to mempool_id`. Disabling
  inductor cudagraph trees removes the capture session while keeping the
  compiled Triton kernels, so there is no measurable throughput loss on eval
  workloads. Callers can opt back in by setting `TORCHINDUCTOR_CUDAGRAPHS`
  explicitly in the adapter environment.
- RoboTwin worker: a non-OOM native evaluator failure with a positive exit
  code is recorded as a terminal failed item (episodes completed before the
  crash are kept and the remainder counted as failures) so the worker continues
  with the next task instead of aborting the whole evaluation pass. OOM and
  signal-terminated exits still restart the worker.

## 0.7.0b5 - 2026-08-14

- RoboTwin launcher: when episode cleanup (`end_episode`) fails while a Policy
  failure is already propagating, keep the original Policy exception as the
  top-level error and drop the cleanup failure instead of letting it replace
  the root cause in the native log tail.
- CALVIN worker: when episode cleanup fails after a failed sequence, keep the
  original `evaluate_sequence` Policy exception and traceback as the top-level
  failure and attach the cleanup error as its diagnostic cause instead of
  letting the cleanup error replace it.

## 0.7.0b3 - 2026-08-14

- Publish the SDK directly to PyPI. The unused TestPyPI gate is removed.
  Runtime is unchanged from 0.7.0b2.

## 0.7.0b2 - 2026-08-13

- RoboTwin runtime patch: select the episode instruction from the per-episode
  seeded `random` stream instead of the unseeded NumPy global RNG, so identical
  seeds yield identical instructions.

## 0.7.0b1 - 2026-08-13

- Remove the retired experimental RoboTwin vectorized runtime from official
  base-image bundles and preserve its design and benchmark report as archived
  documentation.

## 0.6.8b2 - 2026-08-13

- Scale down only the newest evaluator worker after an OOM and retry the
  interrupted work at the reduced concurrency.

## 0.6.8b1 - 2026-08-12

- Preparing the first production PyPI release.

Earlier `0.x` versions were internal image and TestPyPI builds.
