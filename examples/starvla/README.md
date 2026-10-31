# StarVLA Policy example

A `roboscope-kit` `Policy` for StarVLA checkpoints. It is the Policy used for the StarVLA measurements in
the RoboScope Kit technical report; the report's StarVLA listing is a condensed version of
[`starvla_policy.py`](starvla_policy.py). One class serves LIBERO, LIBERO-Plus, RoboTwin, RoboCasa, CALVIN,
and MetaWorld.

## Files

| File | Purpose |
| --- | --- |
| `starvla_policy.py` | The `Policy`: loads one StarVLA run snapshot, preprocesses observations, returns action chunks |
| `starvla_full_checkpoint.py` | Builds the StarVLA framework from `config.yaml` and strictly loads one full checkpoint |
| `starvla_base_assets.py` | Fetches the Qwen base-model metadata the framework needs to build |
| `export_starvla_eval_run.py` | Exports a training run as a self-contained snapshot with relative resource paths |
| `roboscope-policy.json` | Policy process contract, installed at `/workspace/roboscope-policy.json` |

The contract expects these files under `/opt/roboscope/starvla/`, the StarVLA source under
`/root/starVLA`, and a `starVLA` conda environment that has `roboscope-kit` installed.

## Behavior

- `load` declares the benchmark and suite of the session, the features named in `config.yaml`, one
  action-chunk size equal to the checkpoint's `action_horizon`, and an observation stride equal to that
  horizon. On RoboTwin the stride lets the runtime render only at chunk boundaries.
- `act` resizes each camera image to the 224x224 model input the same way the pinned StarVLA client of that
  benchmark does: OpenCV `INTER_AREA` for RoboTwin, RoboCasa, LIBERO-Plus, and MetaWorld; PIL bilinear for
  LIBERO; StarVLA's resize-and-pad for CALVIN. It then unnormalizes the predicted actions with the training
  statistics and applies the configured action transforms.
- The Policy keeps no per-episode state, so `reset` does nothing.

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
action transforms it applies:

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
