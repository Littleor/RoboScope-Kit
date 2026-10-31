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
| [`libero_plus/`](libero_plus/demo.yaml) | [`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1) | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | `985c223` | `starvla_policy.py` |
| [`robotwin/`](robotwin/demo.yaml) | [`StarVLA/Qwen3-VL-OFT-Robotwin2`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2) | `demo_clean`, `demo_randomized` | `7e4a473` | `starvla_policy.py` |
| [`robocasa/`](robocasa/demo.yaml) | [`StarVLA/Qwen3-VL-OFT-Robocasa`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robocasa) | `tabletop` | `985c223` | `starvla_policy.py` |
| [`calvin/`](calvin/demo.yaml) | [`Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D`](https://huggingface.co/Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D) | `task_D_D` | `985c223` | `starvla_policy.py` |
| [`metaworld/`](metaworld/demo.yaml) | [`MINT-SJTU/starvla_metaworld_qwenpiv3_baseline_finetune`](https://huggingface.co/MINT-SJTU/starvla_metaworld_qwenpiv3_baseline_finetune) | `mt50` | `985c223` | `starvla_policy.py` |

StarVLA publishes LIBERO-trained checkpoints for LIBERO-Plus and evaluates them zero-shot; the LIBERO-Plus
demo does the same. StarVLA publishes no CALVIN or MetaWorld checkpoint, so those demos use community
StarVLA checkpoints.

To prepare a demo, check out [StarVLA](https://github.com/starVLA/starVLA) at the listed revision under
`/root/starVLA`, then build the run snapshot:

```bash
/root/miniconda3/envs/starVLA/bin/python /opt/roboscope/starvla/prepare_snapshot.py libero /root/roboscope-checkpoint
```

`prepare_snapshot.py` downloads the checkpoint at the pinned revision and merges the demo's `config` section
into `config.yaml`. Evaluate with the Policy process contract [`roboscope-policy.json`](roboscope-policy.json),
or [`libero/roboscope-policy.json`](libero/roboscope-policy.json) for the LIBERO demo.

## Files

| File | Purpose |
| --- | --- |
| `starvla_policy.py` | The `Policy`: loads one StarVLA run snapshot, preprocesses observations, returns action chunks |
| `starvla_libero_policy.py` | The LIBERO `Policy` of the technical report: follows the StarVLA `eaa51c4` LIBERO client |
| `starvla_full_checkpoint.py` | Builds the StarVLA framework from `config.yaml` and strictly loads one full checkpoint |
| `starvla_base_assets.py` | Fetches the Qwen base-model metadata the framework needs to build |
| `prepare_snapshot.py` | Builds a demo's run snapshot from its `demo.yaml` |
| `export_starvla_eval_run.py` | Exports your own training run as a self-contained snapshot with relative resource paths |
| `roboscope-policy.json` | Policy process contract, installed at `/workspace/roboscope-policy.json` |

The contracts expect these files under `/opt/roboscope/starvla/`, the StarVLA source under
`/root/starVLA`, and a `starVLA` conda environment that has `roboscope-kit` and `huggingface_hub` installed.

## Behavior

- `load` declares the benchmark and suite of the session, the features named in `config.yaml`, one
  action-chunk size equal to the checkpoint's `action_horizon`, and an observation stride equal to that
  horizon. On RoboTwin the stride lets the runtime render only at chunk boundaries.
- `act` resizes each camera image to the 224x224 model input the same way the pinned StarVLA client of that
  benchmark does: OpenCV `INTER_AREA` for RoboTwin, RoboCasa, LIBERO-Plus, and MetaWorld; PIL bilinear for
  LIBERO; StarVLA's resize-and-pad for CALVIN. It then unnormalizes the predicted actions with the training
  statistics and applies the configured action transforms.
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
/root/miniconda3/envs/starVLA/bin/python /opt/roboscope/starvla/export_starvla_eval_run.py \
  --run-dir <training run> \
  --checkpoint <training run>/checkpoints/<selected checkpoint> \
  --output <export directory> \
  --resource-root <root of the local construction resources>
```

The exporter copies the local resources referenced by the `framework` and `inference` sections, rewrites them
to relative paths, and checks that the snapshot loads strictly.
