# X-VLA Policy Example

A `roboscope-kit` `Policy` that runs the public X-VLA-0.9B checkpoints ([2toINF/X-VLA](https://github.com/2toINF/X-VLA),
arXiv:2510.10274) on four benchmarks: calvin, libero, libero_plus, and robotwin. It shows how to write a
stateful Policy and multi-benchmark action contracts. The X-VLA entries on public leaderboards are results recorded by the upstream paper and the community.

## Validation Scope and Research Reference (2026-08-15)

| Benchmark | X-VLA checkpoint used by the demo | Upstream paper/community metric (research reference) | Notes |
|---|---|---|---|
| calvin (task_ABC_D) | `X-VLA-Calvin-ABC_D` | avg_len **4.43** | Community metric available for comparison |
| libero (4 suites) | `X-VLA-Libero` | 98.2/98.6/97.8/97.6, mean **98.1** | Community metric available for comparison |
| libero_plus (4 suites) | `X-VLA-Libero` | No entry | Reported as is; no community reference |
| robotwin (demo_clean/demo_randomized) | `X-VLA-RoboTwin2` | easy **70.0** / hard **39.0** | Directly comparable when the benchmark protocol matches |
| robocasa (tabletop) | — | No entry | **N/A**: the public X-VLA assets cover only the other benchmarks in this table |

The RoboTwin research reference numbers come from the leaderboard settings results table at the pinned
upstream X-VLA commit
[`bcc5152`](https://github.com/2toINF/X-VLA/blob/bcc515295e9286394db607e9fe673b41f25df627/evaluation/robotwin-2.0/README.md).
RoboScope Kit results are directly comparable when the benchmark protocol and cache provenance
match.

## Setup

The example directory and each environment can live anywhere. The Policy reads the upstream code and weights from
`XVLA_ROOT` (by default this example directory, that is, the directory containing `xvla_demo_policy.py`):

- `$XVLA_ROOT/X-VLA`: a checkout of the upstream open-source repository [2toINF/X-VLA](https://github.com/2toINF/X-VLA)
- `$XVLA_ROOT/checkpoints/{calvin,libero,robotwin}`: the `X-VLA-Calvin-ABC_D`,
  `X-VLA-Libero`, and `X-VLA-RoboTwin2` weights downloaded from Hugging Face
- A Python environment (called `xvla` below): python 3.10 + torch 2.1.2 cu121 + transformers 4.51.3 + X-VLA
  requirements + `sapien==3.0.0b1` + `mplib==0.2.1`, with `roboscope-kit` installed

The Policy loads weights only from `$XVLA_ROOT/checkpoints/<benchmark>` and ignores `SessionContext.checkpoint_path`;
pass the same directory to `--checkpoint-source` during evaluation so the result records match the weights actually used.

Build each benchmark's simulation environment with `benchmarks/<benchmark>/build-runtime.sh <install-dir>`, and pass the
`benchmark-manifest.json` in the install directory to `--benchmark-manifest`. The RoboTwin Policy also reads RoboTwin's robot configuration,
so before running, set `ROBOTWIN_RUNTIME_ROOT` to the RoboTwin checkout in that install (the directory that
`runtime.benchmark_root` points to in its `benchmark-manifest.json`).

## Policy Process Contract

[`roboscope-policy.json`](roboscope-policy.json) (schema_version 2) tells the runtime how to start the Policy.
Before use, replace its `/path/to/...` placeholder paths with local paths: `command[0]` is the absolute path to the `xvla` environment's Python,
`command[1]` is the absolute path to `xvla_demo_policy.py`, and `cwd` is an existing absolute directory (the X-VLA checkout here).
The runtime checks these paths before every evaluation.

## Running Evaluations

Run the benchmark CLI from a Python that has `roboscope-kit` installed, for example LIBERO-10:

```bash
export XVLA_ROOT=/path/to/RoboScope-Kit/examples/xvla   # defaults to this directory; can be omitted
python -m roboscope_kit.runtime.benchmark \
    --benchmark libero --suite libero_10 \
    --checkpoint-source "$XVLA_ROOT/checkpoints/libero" \
    --benchmark-manifest /path/to/libero-install/benchmark-manifest.json \
    --policy-config "$XVLA_ROOT/roboscope-policy.json" \
    --output-dir /runs/xvla/libero_10
```

RoboTwin additionally sets `ROBOTWIN_RUNTIME_ROOT`:

```bash
export ROBOTWIN_RUNTIME_ROOT=/path/to/robotwin-install/RoboTwin
python -m roboscope_kit.runtime.benchmark \
    --benchmark robotwin --suite demo_clean \
    --checkpoint-source "$XVLA_ROOT/checkpoints/robotwin" \
    --benchmark-manifest /path/to/robotwin-install/benchmark-manifest.json \
    --policy-config "$XVLA_ROOT/roboscope-policy.json" \
    --output-dir /runs/xvla/demo_clean
```

calvin and libero_plus work the same way (libero_plus uses `checkpoints/libero`). Add `--quick-test` for a bounded smoke run;
`--concurrency K --replicas G` pins K/g, otherwise Autotune selects them. Results are written to `--output-dir` (`result.json`,
`progress.json`, `events.jsonl`, `errors.jsonl`, `autotune/selection.json`, and so on).

## Adaptation Notes (2026-08-15, strict port of the community recipe, no tuning)

- **libero / libero_plus**: matches libero_client.py verbatim: absolute EE actions
  `[pos3, axis-angle3, grip±1]`, 30-step chunks, proprio tracking (real state on the first frame → target at chunk end),
  domain_id=3, steps=10. Proprio rotation uses the **site frame** (= the controller's ee_ori_mat; the observation
  `robot0_eef_quat` is the body frame in xyzw, offset by a constant 90° z rotation, `_libero_site_rotmat`).
  Worker patches (inside the SDK): `use_delta=False` (as stated in the recipe) + MAX_STEPS aligned with
  the recipe's 800/900-step horizons.
- **calvin**: matches calvin_client.py: absolute `(pos, quat_xyzw, grip)` tuples,
  the first 20 rows of the model chunk, domain_id=2, gripper threshold 0.8. The RoboScope Kit calvin spec is a 7-D
  relative action, so this Policy declares
  `calvin.absolute-compact-quat-v1` through `PolicyCapabilities`; the benchmark adapter picks the absolute
  semantics based on the negotiated contract. The adapter first normalizes the unit quat to `w>=0`, then encodes the 3-D xyz; the worker keeps the xyz
  values as is and rebuilds a non-negative w from the unit constraint (tolerating only slight overshoot caused by floating-point error), so the env receives an absolute action
  equivalent to the recipe's.
- **robotwin**: X-VLA is an EE policy (16-D pos+quat+grip), while the RoboScope Kit RoboTwin spec is 14-D
  qpos. This Policy declares `robotwin.ee-compact-quat-v1` through `PolicyCapabilities`,
  and the benchmark adapter picks the EE execution path based on the negotiated contract; the adapter encodes each arm's action as
  `[pos3, canonical quat_xyz3, grip]`, and the launcher rebuilds the full 16-D action and calls the native
  `take_action(..., action_type="ee")`. For the first chunk, proprio uses SAPIEN FK to recover the real
  endpose; later chunks follow the recipe and use the last target pose of the previous chunk, updating the gripper from the current real joints.
  The seed cache still only skips the expert `play_once`, and EE execution keeps the native
  `CuroboPlanner`. StarVLA explicitly declares the qpos contract.
- robocasa declares no capability (X-VLA has no RoboCasa weights; see the scope table).

## Real-Run Validation

- `--quick-test` and other preflights are for bounded smoke runs; full suites provide evidence for comparison with research metrics.
- A full evaluation runs the complete suite defined by the benchmark manifest. `--episodes 1` only reduces the number of episodes per task
  and does not turn the suite into a small sample; for example, LIBERO+ still expands to 10030 variants.
- Pass criteria: CLI exit code 0, `result.json` generated under `--output-dir`, and no terminal failure in `errors.jsonl`
- Metrics are compared with the upstream paper/community X-VLA entries under the same protocol
