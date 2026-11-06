# XPolicyLab Policy example

[`xpolicylab_policy.py`](xpolicylab_policy.py) serves an [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
policy as a `roboscope-kit` `Policy` on the RoboDojo benchmark. XPolicyLab is RoboDojo's policy library: each
`XPolicyLab/policy/<name>/` holds a `Model` with `update_obs`, `get_action`, and `reset`, its `deploy.yml`, and
its own installer. The example loads `XPolicyLab.policy.<name>.model.Model` from the XPolicyLab checkout that
[`benchmarks/robodojo/build-runtime.sh`](../../benchmarks/robodojo/build-runtime.sh) pins, so no model code is
copied here. The RoboDojo integration this builds on is adapted from the work of
[@QLL-CCDD](https://github.com/QLL-CCDD).

Only the Policy is example code; the simulator side is the Kit's built-in RoboDojo benchmark
(`--benchmark robodojo`).

## How it maps

The episode loop is XPolicyLab's reference `deploy.py`: `update_obs` on every control step, `get_action` for a
chunk of action dictionaries, and `update_obs` again between the actions of a chunk. The Policy declares
`step_observation_updates=("robodojo",)`, so the chunk's first observation arrives through `act()` and the
others through `observe()`.

- **Observations.** The canonical features are rebuilt into the fields of RoboDojo's `get_obs()`:
  `vision.cam_head`, `vision.cam_left_wrist`, and `vision.cam_right_wrist` colors, the per-arm joint, gripper,
  and end-effector pose states, and `instruction`. A model that reads anything else (depth, point clouds) is
  not supported.
- **Configuration.** The policy's `deploy.yml` is loaded with the overrides XPolicyLab's server gets for
  RoboDojo (`bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `seed=1`, `action_type`), and
  `--checkpoint-source` is passed as `checkpoint_path`. `task_name` is not set, because one replica serves
  every task.
- **Actions.** `--action-type joint` returns `robodojo.x5-joint-v1`; `--action-type ee` converts each
  `[x, y, z, qw, qx, qy, qz]` pose to `robodojo.x5-ee-compact-quat-v1`. The default is the policy's
  `deploy.yml` `action_type`.
- **Episode state.** A model that scopes its state by evaluation ID (it has `reset_evaluation`, as
  `Simate_beta` does) gets the kit's `episode_id` as that ID, so one replica can serve several Workers.
  Any other model keeps one episode of state; the Policy refuses a second concurrent episode, so run it with
  as many replicas as Workers: `--concurrency K --replicas K`.

## Run Simate_beta

Use a new or empty `--output-dir` for each quick test, self-test, or evaluation. Only directories
containing `errors.jsonl` alone can be retried in place. Existing run artifacts are preserved, not resumed.
Keep shell and scheduler logs outside the output directory.

`Simate_beta` is the validated configuration: the released Simate-beta AE checkpoint
([`SIPAILab/sipai-robodojo-eval-ae`](https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae) at `0738aab`),
FP32, 10 flow steps, 50 predicted actions of which the first 10 are executed, and a 25-frame history sampled
every 20 steps. It is a public RoboDojo leaderboard submission by SIPAILab, not a checkpoint trained by the RoboDojo
authors. The paths below are placeholders.

1. Build the simulator. Read the
   [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
   first:

   ```bash
   export OMNI_KIT_ACCEPT_EULA=YES
   benchmarks/robodojo/build-runtime.sh /sim/robodojo
   ```

2. Create the policy environment with the policy's own installer, using the build's Conda, and add
   `roboscope-kit`:

   ```bash
   cd /sim/robodojo/RoboDojo/XPolicyLab/policy/Simate_beta
   CONDA_EXE=/sim/robodojo/miniforge/bin/conda bash install.sh simate-beta-eval
   /sim/robodojo/miniforge/envs/simate-beta-eval/bin/python -m pip install /path/to/RoboScope-Kit/packages/roboscope-kit
   ```

   The build's Conda uses `conda-forge`. Where that is slow or unreachable, set `CONDA_CHANNELS` to a reachable
   channel for this command; a `CONDARC` file alone does not replace it.

3. Download the checkpoint with the policy's own script, which checks the SHA-256 of every file:

   ```bash
   PYTHON=/sim/robodojo/miniforge/envs/simate-beta-eval/bin/python \
     bash download_checkpoint.sh /models/simate-beta-robodojo-eval
   ```

4. Copy [`roboscope-policy.json`](roboscope-policy.json), set its paths, and run. `cwd` must be the RoboDojo
   checkout, which contains XPolicyLab:

   ```bash
   /sim/robodojo/env/bin/python -m roboscope_kit.runtime.benchmark \
       --benchmark robodojo --suite arx_x5 \
       --checkpoint-source /models/simate-beta-robodojo-eval \
       --benchmark-manifest /sim/robodojo/benchmark-manifest.json \
       --policy-config /path/to/roboscope-policy.json \
       --output-dir /runs/simate-arx_x5 --quick-test
   ```

   `--quick-test` runs one episode of the suite's first task, `align_blocks`. It shows that the Policy and
   simulator work together; the episode itself may fail, because outcomes depend on the task and vary between
   runs. Without it, the run is the full `arx_x5` suite.

   Loading the Policy shares the manifest's `ready_timeout_s`. If a first load from slow storage ends in
   `TimeoutError`, raise that value in `benchmark-manifest.json`, or run again once the checkpoint is cached.

Another XPolicyLab policy runs the same way with `--policy <name>` in `roboscope-policy.json`, its own
environment and checkpoint, and, unless it has `reset_evaluation`, `--concurrency K --replicas K`.

## Evaluation protocol

- **Tasks.** `--suite arx_x5` is RoboDojo's 42 base tasks; `arx_x5_randomized` adds 12 randomized variants. Each
  task has 25 or 50 fixed layouts. `--episodes N` evaluates the first N layouts of every task with seed 1, the
  same layouts as upstream `EVAL_NUM=N`.
- **Episodes.** One episode per layout. It ends when RoboDojo's success check passes or at the task's step limit,
  which is between 200 and 1,900 control steps. RoboDojo also gives each episode a partial-credit score from 0 to
  1. An episode takes from under a minute to about ten minutes.
- **Results.** A task's success rate is over its episodes; the overall rate and mean score are over all episodes.
  The Kit rejects a layout shard that RoboDojo returns incomplete and evaluates it again, so a result has exactly
  one record per task and layout.

## Results

Simate_beta, layouts 0–2 of all 42 `arx_x5` tasks (126 episodes), on one RTX 4090 48 GB, 2026-11-06. The Kit
(0.8.0b25) ran with its defaults, which autotuning set to two simulator Workers sharing one Policy replica; upstream
ran its own client with one environment, task by task.

| | Kit | Upstream native |
|---|---:|---:|
| Successes | 26/126 (20.6%) | 26/126 (20.6%) |
| Mean score | 0.278 | 0.271 |
| Evaluation time | 4.4 h (15,770 s) | 6.6 h (23,753 s) |
| Total time, including the Kit's 30 min autotuning | 4.9 h (17,576 s) | 6.6 h (23,753 s) |

**No difference between the Kit and upstream was found.** Paired by task and layout, 6 episodes succeed only in
the Kit and 6 only upstream (McNemar p = 1.0), and 88 of the 94 episodes that fail on both sides stop at the same
step. The same layout can succeed in one run and fail in the next, so expect differences of an episode or two per
task between any two runs.

Throughput tuning is tracked in [#18](https://github.com/MINT-SJTU/RoboScope-Kit/issues/18). Per-episode records
and timings are in the [raw data](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/robodojo-simate-acceptance-20261106);
discussion is in [#17](https://github.com/MINT-SJTU/RoboScope-Kit/issues/17).

## Known limitations

- Only `Simate_beta` has been compared with upstream. Other XPolicyLab policies use the same mapping.
- RoboDojo episodes are not reproducible run to run, so compare runs statistically rather than episode by episode.
