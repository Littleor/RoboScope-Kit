# RoboScope Kit Profiling

Updated: 2026-11-05

This document explains how to use `--profile` to see where an evaluation's time goes, and records what
profiling has found so far. After every profiling run, add its findings to the benchmark's own documentation and link them from
[Findings](#findings), so results accumulate instead of being measured again from scratch.

## Requirements

- **Nsight Systems CLI.** The machine that runs the evaluation needs
  [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) with `nsys` on `PATH`. The Kit
  does not install it. Version 2026.5.1 is verified (installed under `/opt/nvidia/nsight-systems-cli/2026.5.1`
  in the AutoDL container). Without `nsys`, `--profile` stops before any Worker starts.
- **No extra Python package.** NVTX ranges are written with `ctypes` through Nsight Systems' own injection
  library (see [How it works](#how-it-works)), so neither the simulator environment nor the Policy
  environment needs `nvtx`. Without `--profile`, nothing is loaded.
- **A fixed configuration.** `--profile` works only with `--quick-test` or a fixed
  `--concurrency K --replicas G`; autotune probes are never captured.
- **Disk space.** Each Worker writes one report, roughly proportional to the episode's steps: on RoboDojo,
  about 27 MB for a 20-step episode and about 150 MB for a 105-step episode. `nsys` also writes temporary
  files under `/tmp` while capturing.

## Running

```bash
python -m roboscope_kit.runtime.benchmark --benchmark robodojo --suite arx_x5 \
  --checkpoint-source <ckpt> --benchmark-manifest <install-dir>/benchmark-manifest.json \
  --policy-config <policy.json> --output-dir <out> \
  --quick-test --profile
```

To look at concurrent Workers, replace `--quick-test` with `--concurrency K --replicas G`.

## Output

- `<out>/profile/worker-<id>-attempt-<n>.nsys-rep`: the first episode of each Worker, with the CUDA kernels
  and API calls of the Worker and its child processes (such as the Isaac Sim evaluator), and NVTX ranges:
  - `episode` (domain `roboscope`): from `PolicyClient.reset` to `end_episode`; this is the capture range.
  - `policy.<operation>`: each Policy round trip, such as `policy.act`, `policy.observe`, and
    `policy.reset`.
- `<out>/manifest.json` records `"profile": true`. Tracing slows the run, so its throughput is not a
  measurement.

## Reading a report

- **GUI:** open the `.nsys-rep` in Nsight Systems and inspect the timeline.
- **Command line:** `nsys stats <report>.nsys-rep`.
- **Custom metrics:** export to SQLite and query it. This script prints the total time of each NVTX range
  and the GPU busy time (the union of all kernel intervals):

```bash
nsys export --type=sqlite -o report.sqlite worker-000-attempt-000.nsys-rep
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("report.sqlite")
for row in c.execute("""select coalesce(n.text, s.value), count(*), sum(n.end - n.start) / 1e9
                        from NVTX_EVENTS n left join StringIds s on n.textId = s.id
                        where n.end is not null group by 1 order by 3 desc"""):
    print(row)
busy, covered = 0, 0
kernels = c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
for start, end in kernels:
    start = max(start, covered)
    if end > start:
        busy, covered = busy + end - start, end
print("kernels", len(kernels), "busy s", busy / 1e9, "window s", (kernels[-1][1] - kernels[0][0]) / 1e9)
PY
```

## Coverage

Capture starts when the Worker's `PolicyClient` opens the `episode` range, so only benchmarks that call the
Policy through `PolicyClient` record anything. The Policy Server does not run under `nsys`: its own GPU time
is not in the report, only the round-trip time of each request.

| Benchmark | Uses `PolicyClient` | Verified on hardware |
|---|---|---|
| RoboDojo | Yes | Yes (2026-11-05, K=1 and K=3) |
| LIBERO, LIBERO-Plus | Yes | Not yet |
| MetaWorld | Yes | Not yet |
| robomimic | Yes | Not yet |
| RoboCasa | Yes | Not yet |
| CALVIN | Yes | Not yet |
| RoboTwin | No, its launcher has its own client | Records nothing |

## How it works

- The runtime wraps each Worker's command in `nsys profile` with `--trace=cuda,nvtx --sample=none
  --cpuctxsw=none --capture-range=nvtx --nvtx-capture=episode@roboscope --capture-range-end=stop`, so only
  the first episode is captured and no time window has to be guessed.
- When `nsys` launches a process, it names its injection library in `NVTX_INJECTION64_PATH`.
  `roboscope_kit.profiling` loads that library with `ctypes`, follows the same initialization as NVTX's
  header-only C API (`InitializeInjectionNvtx2` and the export tables) to obtain `nvtxDomainCreateA`,
  `nvtxDomainRangeStartEx`, and `nvtxDomainRangeEnd`, and writes the ranges with them.
- Ranges are written only when `ROBOSCOPE_PROFILE` is set and the process actually runs under `nsys`.

## Findings

Each benchmark keeps its profiling findings in its own documentation, newest first, with the date, machine,
configuration, and Kit version:

- [RoboDojo](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robodojo/README.md#profiling):
  `arx_x5` keeps the GPU busy only about 11% of an episode (2026-11-05).
