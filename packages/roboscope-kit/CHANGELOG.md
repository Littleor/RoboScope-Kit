# Changelog

## 0.8.0b1 (Beta, published 2026-10-30)

- Rename the SDK to RoboScope Kit. Install it with `pip install roboscope-kit` (previously
  `roboscope-inference-sdk`) and import it as `roboscope_kit` (previously `roboscope_inference_sdk`).
  `importlib.metadata.version()` now takes `"roboscope-kit"`. No compatibility alias is provided.
- Move the package source from `packages/roboscope-inference-sdk/` to `packages/roboscope-kit/`; image
  Control Runtimes now live under `/opt/roboscope/control-runtime/packages/roboscope-kit/`.

## 0.7.0b77 (Unreleased)

- Add an opt-in CUDA MPS mode for Policy replicas: `--cuda-mps` on the
  benchmark CLI and `run-benchmark.sh`, off by default. The runtime owns a
  private MPS control daemon for the run, attaches every replica to it,
  fails if a replica does not attach, terminates a replica's MPS clients
  before signalling it, and records the session in `result.json`. Autotune
  probes run under the same mode as the formal pass.

## 0.7.0b76 (Unreleased)

- Revert the MetaWorld runtime, autotune, native-reference and MPS changes
  because they regressed the autotune algorithm. Restore the b59 runtime
  and autotune behavior.

## 0.7.0b59 (Unreleased candidate)

- RoboTwin renders one observation per refresh. The evaluator loop now asks the
  Policy client whether an observation is due, so steps at chunk or stride
  boundaries no longer render twice and stride 1 matches the official cadence.
- Keep the per-step observation cadence under randomized lighting, the viewer,
  video logging and point-cloud observations, whose output or RNG stream depends
  on every render.
- The StarVLA official Policy declares its action horizon as its observation
  stride, replacing the external stride-16 launcher wrapper.

## 0.7.0b58 (Unreleased candidate)

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

## Unreleased

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
