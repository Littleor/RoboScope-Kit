# RoboTwin Scanner and SDK Evaluation Performance Report

Date: 2026-07-22

## Scope

This report validates two changes to the unified benchmark runtime:

1. `roboscope_robotwin_seed_scan.py` validates RoboTwin seeds with a shared
   SAPIEN engine/renderer and persistent cuRobo planners.
2. The unified RoboScope SDK RoboTwin evaluation consumes the Scanner cache and
   skips the per-evaluation expert `play_once` path.

The measurements use `adjust_bottle` with the `demo_clean` task configuration.

## Test System

- CPU: 208 logical CPUs
- GPU: 4 x NVIDIA GeForce RTX 4090, 49,140 MiB each
- Scanner SHA-256:
  `e049a7c25495d491a8eaa5bd208be1aaa9f645ecc49f757d3bb8e7ecba25029c`

The SHA-256 of the Scanner executed on the test system matched the repository
file exactly.

## Scanner Throughput

### Single process

The first seed includes one-time renderer and cuRobo initialization. Steady-state
metrics exclude that first seed.

| Metric | Result |
| --- | ---: |
| Seeds scanned | 6 |
| Valid seeds | 6 |
| First setup | 25.04 s |
| Steady setup per seed | 1.008 s |
| Steady `play_once` per seed | 5.174 s |
| Steady time per seed | 6.182 s |
| Steady throughput | 9.7 seeds/minute |

### Four GPUs, 32 Scanner processes

Eight Scanner processes were assigned to each GPU. Each process scanned eight
seeds. Process startup was staggered to avoid simultaneous cuRobo warmup.

| Metric | Result |
| --- | ---: |
| Processes completed | 32 / 32 |
| Seeds scanned | 256 |
| Valid seeds | 210 |
| Valid rate | 82.0% |
| Processes with traceback | 0 |
| Per-process steady throughput | 2.4-5.1 seeds/minute |
| Aggregate steady throughput | 105.0 seeds/minute |

The aggregate result is calculated by summing each completed process's reported
`steady_seeds_per_min`. At this concurrency, CPU contention reduces each process
from 9.7 seeds/minute to 2.4-5.1 seeds/minute.

### CPU thread oversubscription

The scanner now defaults `--cpu-threads` to `1`. It configures Torch intra-op
and inter-op threads together with OpenMP, MKL, OpenBLAS, and NumExpr before
RoboTwin or cuRobo is imported. The option remains explicit so a deployment can
measure a different positive value instead of relying on the host CPU count.

The same `adjust_bottle` seeds were scanned before and after the limit. Native
thread peaks were sampled from `/proc/<pid>/status`, and every paired cache file
was byte-identical.

| Scanner processes | Metric | Host default | One CPU thread |
| ---: | --- | ---: | ---: |
| 1 | Peak native threads | 346 | 21 |
| 1 | Steady throughput | 10.2 seeds/min | 10.1 seeds/min |
| 8 | Peak native threads, total | 2,775 | 117 |
| 8 | Wall time, 32 seeds | 76 s | 73 s |
| 32 | Peak native threads, total | 10,852 | 450 |
| 32 | Wall time, 96 seeds | 164 s | 149 s |
| 32 | End-to-end throughput | 35.1 seeds/min | 38.7 seeds/min |

At 32 processes the limit reduced the sampled native-thread peak by 95.9%,
reduced wall time by 9.1%, and improved end-to-end throughput by 10.1%. It does
not remove the remaining cuRobo, GPU, expert `play_once`, and physics contention;
its primary benefit is preventing thousands of unnecessary host threads without
changing seed validity.

The requested 500 seeds/minute target was not achieved. The measured result is
21% of that target. On this host, adding processes beyond 32 is not expected to
scale linearly because expert `play_once` and physics simulation are CPU-heavy.

## Unified SDK End-to-End Evaluation

The end-to-end check used the public unified entrypoint:

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark robotwin \
  --suite demo_clean \
  --checkpoint-source <checkpoint> \
  --output-dir <output> \
  --quick-test \
  --image-manifest <manifest>
```

Both measurements used one GPU, one SDK Policy replica, one RoboTwin worker,
video disabled, a 16-action-chunk hold policy, and one 400-step episode. This is
a runtime-path benchmark, not a policy-quality benchmark.

### First-run comparison

| Metric | Expert seed path | Scanner cache path | Change |
| --- | ---: | ---: | ---: |
| Total SDK E2E time | 176.40 s | 105.52 s | -40.2% |
| SDK E2E throughput | 2.27 steps/s | 3.79 steps/s | 1.67x |
| Episode body time | 33.20 s | 28.39 s | -14.5% |
| Worker failures | 0 | 0 | unchanged |
| Tracebacks | 0 | 0 | unchanged |

The cache-path record used seed `1000060`, which came directly from the Scanner
output. The evaluation completed all 400 steps through the unified SDK adapter,
worker, launcher, and Policy socket path.

The main E2E reduction comes from removing the expert seed-selection work and
cuRobo initialization from evaluation startup. The Scanner performs that work
once ahead of evaluation and stores both the valid seed and its `episode_info`.

### Warm-run observation

After the system had already initialized GPU, renderer, shader, and filesystem
caches, the required-cache path completed in 46.35 seconds, or 8.63 steps/s.
The episode body took 25.21 seconds. This warm-run value is recorded only as an
observation and is not the formal performance comparison because there is no
paired warm expert-path measurement.

## Integration Guarantee

The new SDK RoboTwin entrypoint requires the Scanner cache for full evaluation
runs while allowing image smoke tests to run before a cache has been generated:

- Full evaluation requires `seed_cache_dir` in the benchmark image manifest.
- `--quick-test` and image `--self-test` may omit `seed_cache_dir`.
- A quick-test without cache uses the official expert `play_once` path; when a
  cache is configured it uses the same optimized cache path as a full run.
- When configured, the worker validates the task cache entry count and its
  `seed` / `episode_info` structure before launching RoboTwin.
- The launcher installs the evaluation-only gripper planner only when
  `ROBOTWIN_SEED_CACHE_PATH` is present.
- Missing, incomplete, or malformed configured caches fail explicitly.

A full-run request without `seed_cache_dir` was verified to fail during factory
validation. A quick-test without cache constructs a worker command without
`--seed-cache-dir`. A manifest with the Scanner cache completed the full SDK
end-to-end path with zero worker failures and zero tracebacks.

## Conclusions

- The Scanner is connected to the unified SDK RoboTwin evaluation path.
- The cache is mandatory for full evaluation, while quick-test and self-test can
  run without it for image validation.
- First-run E2E throughput improved from 2.27 to 3.79 steps/s (1.67x).
- Scanner throughput reached 105.0 seeds/minute on 4 GPUs and 32 processes.
- The measurements do not demonstrate 400 evaluation steps/s or 500 seed
  validations/minute. Those targets remain unmet on the tested host.

## Old Evaluator Paired Comparison

The final migration check used the existing `grab_roller` `demo_clean`
paired-50 cache and the StarVLA `steps_40000_pytorch_model.pt` checkpoint. The
old evaluator and unified runtime consumed the same 50 cached seeds and episode
metadata with one worker, one policy replica, one GPU, DDIM with 10 steps,
bf16, and a 16-action chunk.

The comparison exposed two correctness bugs in the optimized control loop:

- mplib TOPP failures terminated the unified worker, while the official
  evaluator continues for 50 control iterations without updating the failed
  arm and still interpolates its gripper;
- the hoisted gripper implementation calculated drive velocity from the
  normalized gripper value instead of the mapped joint drive target.

Both paths now match the official evaluator semantics. The seed that originally
triggered `RuntimeError("Fail to parameterize path")`, `100025`, completed after
the fix with the same old/new result: failure at 400 policy steps.

| Metric | Old evaluator | Unified runtime |
| --- | ---: | ---: |
| Paired seed coverage | 50 / 50 | 50 / 50 |
| Successes | 47 | 47 |
| Success rate | 94.0% | 94.0% |
| Failure seeds | 100025, 100028, 100035 | 100025, 100028, 100035 |
| Policy steps | 5,020 | 4,998 |
| Sum of native episode time | 641.50 s | 319.20 s |
| End-to-end runtime time | not recorded | 334.91 s |
| Worker failures | 0 | 0 |

All 50 outcomes matched and there were no success flips. Terminal step counts
matched exactly for 17 episodes; the remaining 33 differed by -10 to +3 steps,
with an aggregate delta of -22 steps (-0.44%). This residual is not a model or
seed mismatch. Replaying the first old observation directly through the current
StarVLA wrapper produced a bit-exact 16-action chunk (`max_abs_diff = 0`). The
current evaluator's initial rendered observations differed slightly from the
archived old trace before any action was applied: mean absolute RGB differences
were 0.038 for the head camera and 0.026/0.028 for the wrist cameras on a
0-255 scale. Wrist differences were at most one intensity level; the head
camera had a small number of pixels with a maximum difference of 13. Repeating
the run with 208 client CPU threads instead of one produced the same terminal
steps, so the unified thread limit was not the cause.

The paired acceptance criterion for the evaluator migration is therefore met:
complete paired coverage, identical aggregate score, identical per-seed
outcomes, no worker recovery, and an explained sub-percent terminal-step delta
caused by archived-versus-current rendered pixels rather than evaluation logic.
