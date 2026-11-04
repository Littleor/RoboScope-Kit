# XPolicyLab Policy example

[`xpolicylab_policy.py`](xpolicylab_policy.py) serves an [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
policy as a `roboscope-kit` `Policy` on the RoboDojo benchmark. XPolicyLab is RoboDojo's policy library: each
`XPolicyLab/policy/<name>/` holds a `Model` with `update_obs`, `get_action`, and `reset`, its `deploy.yml`, and
its own installer. The example loads `XPolicyLab.policy.<name>.model.Model` from the XPolicyLab checkout that
[`benchmarks/robodojo/build-runtime.sh`](../../benchmarks/robodojo/build-runtime.sh) pins, so no model code is
copied here. The RoboDojo integration this builds on is adapted from the work of
[@QLL-CCDD](https://github.com/QLL-CCDD).

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

`Simate_beta` is the validated configuration: the released Simate-beta AE checkpoint
([`SIPAILab/sipai-robodojo-eval-ae`](https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae) at `0738aab`),
FP32, 10 flow steps, 50 predicted actions of which the first 10 are executed, and a 25-frame history sampled
every 20 steps. The paths below are placeholders.

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
