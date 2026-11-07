# Benchmark Integration Acceptance

This standard applies to a new benchmark integration and to a new suite or protocol for an existing
benchmark. A draft can be reviewed at any point; acceptance requires all of the gates below. A
partial run, an implementation-only PR, or a similar overall score does not by itself pass acceptance.

## 1. Freeze the scope and protocol

- Pin the upstream benchmark, simulator, assets, model and Kit commits or versions, and record the
  immutable source and checksum of each external input. Identify any license or download prerequisite.
- Enumerate every upstream evaluation task and variant in the requested benchmark or suite at that
  revision. For a platform without a single official suite, inventory its registered environments,
  distinguish evaluation tasks from helper/example environments with evidence, and include every
  evaluation task. Any missing task remains an explicit blocker; a smaller phase must be named as a
  partial integration and cannot be presented as the full benchmark.
- Put the complete task/variant list in a versioned manifest. For every supported profile record the
  robot, observation and action spaces, cameras, controller and frequencies, action horizon, episode
  count, seeds or initial states, time limit, termination and success rules, and aggregation method.
  Use the official protocol where one exists; otherwise name the chosen protocol and its source.
- Record the intended checkpoint or policy for every task and supported observation profile. A task
  without a runnable policy and native comparison is not yet validated.

## 2. Reproduce the environment

- Build in a fresh directory with the committed `benchmarks/<benchmark>/build-runtime.sh` and its pin
  files. Run the Kit `--self-test` and a real-policy `--quick-test`; record the commands, hardware,
  runtime versions and results. Do not repair the installed environment by hand.
- Run the repository's required checks and obtain passing CI for the exact PR head. Pending,
  action-required or missing checks are not passing checks. Identify any skipped check and why.

## 3. Run every task

- Run the full frozen matrix with a real policy: every task, variant and claimed observation profile,
  using the prescribed episode counts and seeds. A quick test or representative subset is supporting
  evidence only. Record expected and actual counts per task and profile.
- Require one terminal result for each expected episode, with no missing, duplicate, unknown or
  substituted records. Record failures and retries. A partial or individual shard must not be marked
  `formal`; a merged run is complete only after coverage and identity are checked across all shards.
- Save the per-episode identity and result, per-task metrics, full configuration, logs and commands.
  Report the exact Kit head and whether each result came from that head. After a relevant code,
  protocol or dependency change, rerun the affected evidence rather than relabeling an older run.

## 4. Compare with the native evaluator

- Compare every task against the pinned native evaluator using the same model, task/variant, episode
  identities, initial conditions, observations, actions, termination rules and scoring. Record any
  unavoidable difference in hardware or environment. The comparison must check these inputs, reject
  duplicate, unknown, missing and unpaired records, and fail acceptance if coverage is incomplete.
- Report per-task success counts and rates, paired outcome disagreements, executed steps and the
  first observation/action divergence where available. An aggregate score can hide opposite changes
  between tasks and is not evidence of per-task agreement.
- Establish repeatability on the native path and the Kit path. For a deterministic protocol, investigate
  mismatched observations, actions or outcomes. For a nondeterministic protocol, state the sample size,
  comparison method and acceptable tolerance before collecting the acceptance runs; quantify native
  repeat variation and explain remaining per-task differences. Do not claim equivalence merely because
  two single-run totals are close.

## 5. Deliver reviewable evidence

- Link durable raw records, logs, manifests, comparison reports and exact reproduction commands from
  the PR. Include task-by-task coverage, exclusions/blockers, resource configuration and the final
  acceptance decision. Session-local files alone are not a durable handoff.
- Keep the benchmark README, package README and protocol document consistent with the implemented and
  validated scope. A PR's evidence must identify its exact head; do not imply that a draft, an old-head
  result or a smoke test validates the current full integration.

## Performance work

Integration establishes a correctness and performance baseline; an optimization is accepted against
the target in its own issue. Compare baseline and candidate over the same full workload, checkpoint,
initial conditions, observation/action and scoring protocol, output settings and equivalent hardware
resources. Record configuration changes, including concurrency and replicas. Report formal evaluation
and end-to-end time, throughput, per-task outcomes, resource usage and variability across repeated runs.
Do not shorten episodes, omit tasks or change observations to achieve a speedup. A target such as the
2x requirement in RoboCasa #16 applies to that optimization, not automatically to every integration.
