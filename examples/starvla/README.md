# StarVLA Policy examples

`roboscope-kit` `Policy` code for StarVLA checkpoints, with one runnable demo per benchmark. The LIBERO and
RoboTwin demos are the configurations behind the StarVLA results in the RoboScope Kit technical report; the
report's StarVLA listing is a condensed version of [`starvla_policy.py`](starvla_policy.py).

## Demos

Each demo directory holds a `demo.yaml` that names a published checkpoint, the StarVLA revision to run it
with, the Policy file, and the configuration merged into the checkpoint's `config.yaml`.

| Demo | Checkpoint | Suites | StarVLA revision | Policy |
| --- | --- | --- | --- | --- |
| [`libero/`](libero/demo.yaml) | [`StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1`](https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1) | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | `7e4a473` | `starvla_libero_policy.py` |
| [`libero_pro/`](libero_pro/demo.yaml) | [`StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1`](https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1) | the 16 `libero_{goal,spatial,10,object}_{object,swap,lan,task}` suites | `7e4a473` | `starvla_policy.py` |
| [`libero_plus/`](libero_plus/demo.yaml) | [`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1) | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | `985c223` | `starvla_policy.py` |
| [`robotwin/`](robotwin/demo.yaml) | [`StarVLA/Qwen3-VL-OFT-Robotwin2`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2) | `demo_clean`, `demo_randomized` | `7e4a473` | `starvla_policy.py` |
| [`robocasa/`](robocasa/demo.yaml) | [`StarVLA/Qwen3-VL-OFT-Robocasa`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robocasa) | `tabletop` | `985c223` | `starvla_policy.py` |
| [`calvin/`](calvin/demo.yaml) | [`Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D`](https://huggingface.co/Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D) | `task_D_D` | `985c223` | `starvla_policy.py` |
| [`metaworld/`](metaworld/demo.yaml) | [`MINT-SJTU/starvla_metaworld_qwenpiv3_baseline_finetune`](https://huggingface.co/MINT-SJTU/starvla_metaworld_qwenpiv3_baseline_finetune) | `mt50` | `985c223` | `starvla_policy.py` |

StarVLA publishes LIBERO-trained checkpoints for LIBERO-Plus and evaluates them zero-shot; the LIBERO-Plus
demo does the same. The LIBERO-Pro demo evaluates the LIBERO demo's checkpoint zero-shot with
`starvla_policy.py`, which preprocesses like the LIBERO client of StarVLA `7e4a473`, the native evaluator it
was compared against. StarVLA publishes no CALVIN or MetaWorld checkpoint, so those demos use community
StarVLA checkpoints.

## Run a demo

Use a new or empty `--output-dir` for each quick test, self-test, or evaluation. Only directories
containing `errors.jsonl` alone can be retried in place. Existing run artifacts are preserved, not resumed.
Keep shell and scheduler logs outside the output directory.

The example directory, the StarVLA checkout, and every environment can live wherever you choose; the paths
below are placeholders.

1. Check out [StarVLA](https://github.com/starVLA/starVLA) at the demo's revision, for example in
   `/path/to/starVLA`, and create a Python environment for it (called `starVLA` below) with StarVLA's
   requirements, `roboscope-kit`, and `huggingface_hub`.
2. Build the benchmark's simulator environment with
   [`benchmarks/<benchmark>/build-runtime.sh <install-dir>`](https://github.com/MINT-SJTU/RoboScope-Kit/tree/main/benchmarks);
   it copies the benchmark manifest to `<install-dir>/benchmark-manifest.json`.
3. Build the run snapshot. `prepare_snapshot.py` downloads the checkpoint at the pinned revision, merges
   the demo's `config` section into `config.yaml`, and downloads the Qwen base model's configuration and
   tokenizer (no weights) from Hugging Face into the path `config.yaml` names:

   ```bash
   /path/to/envs/starVLA/bin/python /path/to/RoboScope-Kit/examples/starvla/prepare_snapshot.py \
     libero /models/starvla-libero
   ```

4. Edit the Policy process contract: [`libero/roboscope-policy.json`](libero/roboscope-policy.json) for the
   LIBERO demo, [`roboscope-policy.json`](roboscope-policy.json) for every other demo. Replace the
   `/path/to/...` placeholders: `command[0]` is the absolute path of the `starVLA` environment's Python,
   `command[1]` the absolute path of the Policy file in this directory, and `cwd` the StarVLA checkout. The
   runtime checks that `command[0]` is an executable file and `cwd` an existing directory before every run.
5. Evaluate with the benchmark CLI, from any Python environment that has `roboscope-kit` installed. The run
   snapshot is the checkpoint source:

   ```bash
   python -m roboscope_kit.runtime.benchmark \
       --benchmark libero --suite libero_10 --checkpoint-source /models/starvla-libero \
       --benchmark-manifest /path/to/libero-install/benchmark-manifest.json \
       --policy-config /path/to/RoboScope-Kit/examples/starvla/libero/roboscope-policy.json \
       --output-dir /runs/starvla/libero_10
   ```

   Add `--quick-test` for a short smoke run, or `--concurrency K --replicas G` to evaluate at a fixed K/g
   instead of letting Autotune choose. Results, progress, and logs are written under `--output-dir`.

The Policy files import each other, so keep them together in this directory. They import StarVLA from the
Policy's working directory (the contract's `cwd`); set `STARVLA_ROOT` to use another checkout. The Policy
loads offline and stops if the snapshot lacks its base-model metadata; for a snapshot you exported yourself,
fetch it with `python starvla_base_assets.py <run-snapshot>`.

## Files

| File | Purpose |
| --- | --- |
| `starvla_policy.py` | The `Policy`: loads one StarVLA run snapshot, preprocesses observations, returns action chunks |
| `starvla_libero_policy.py` | The LIBERO `Policy` of the technical report: follows the StarVLA `eaa51c4` LIBERO client |
| `starvla_full_checkpoint.py` | Builds the StarVLA framework from `config.yaml` and strictly loads one full checkpoint |
| `starvla_base_assets.py` | Fetches the Qwen base-model metadata the framework needs to build, from Hugging Face |
| `prepare_snapshot.py` | Builds a demo's run snapshot from its `demo.yaml` |
| `export_starvla_eval_run.py` | Exports your own training run as a self-contained snapshot with relative resource paths |
| `roboscope-policy.json` | Policy process contract, passed to the runtime with `--policy-config` |
| `libero/roboscope-policy.json` | Policy process contract of the LIBERO demo |


## Behavior

- `load` declares the benchmark and suite of the session, the features named in `config.yaml`, one
  action-chunk size equal to the checkpoint's `action_horizon`, and an observation stride equal to that
  horizon. On RoboTwin the stride lets the runtime render only at chunk boundaries.
- `act` resizes each camera image to the 224x224 model input the same way the pinned StarVLA client of that
  benchmark does: OpenCV `INTER_AREA` for RoboTwin, RoboCasa, LIBERO-Plus, and MetaWorld; PIL bilinear for
  LIBERO and LIBERO-Pro; StarVLA's resize-and-pad for CALVIN. It then unnormalizes the predicted actions with
  the training statistics and applies the configured action transforms.
- `starvla_libero_policy.py` instead resizes LIBERO images with OpenCV `INTER_AREA`, clips the normalized
  actions to [-1, 1], binarizes the gripper, and then unnormalizes with the min/max statistics.
- Neither Policy keeps per-episode state, so `reset` does nothing.

## Run snapshot

`SessionContext.checkpoint_path` points to a StarVLA run directory, or to the checkpoint inside it:

```text
<run>/
├── config.yaml
├── dataset_statistics.json
├── checkpoints/
│   └── <one full checkpoint, .pt or .safetensors>
└── <local resources referenced by config.yaml>
```

`config.yaml` must carry an `inference` section that names the observation features the Policy reads and the
action transforms it applies; the demo files show one per benchmark:

```yaml
inference:
  dtype: bfloat16                    # required: bfloat16, float16 or float32
  unnorm_key: franka                 # optional: training normalization statistics key
  inputs:                            # required
    images:                          # required, non-empty; order is the camera order
      - images.primary
      - images.wrist
    language: language.instruction   # required
    state: state.proprioception      # optional, only if the model reads state
  kwargs:                            # optional, forwarded to predict_action
    do_sample: false
  action_transforms:                 # optional, applied in order
    - index: 6
      operation: threshold
      threshold: 0.5
      above: -1.0
      below: 1.0
```

`action_transforms` supports `threshold` (`index`, `threshold`, `above`, `below`) and `affine` (`index`,
`scale`, `offset`). The example reproduces the LIBERO gripper binarization. Camera order and transforms must
match training; the runtime cannot detect a wrong but well-formed choice.

A training run usually references resources by absolute paths on the training host. Export it before
evaluation:

```bash
STARVLA_ROOT=/path/to/starVLA /path/to/envs/starVLA/bin/python \
  /path/to/RoboScope-Kit/examples/starvla/export_starvla_eval_run.py \
  --run-dir <training run> \
  --checkpoint <training run>/checkpoints/<selected checkpoint> \
  --output <export directory> \
  --resource-root <root of the local construction resources>
```

The exporter copies the local resources referenced by the `framework` and `inference` sections, rewrites them
to relative paths, and checks that the snapshot loads strictly.
