# robomimic model-zoo Policy

A `roboscope-kit` `Policy` that serves the five robomimic v0.1 proficient-human (PH) low-dimensional
BC-RNN checkpoints of the [robomimic model zoo](https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html)
on the `robomimic` benchmark's `ph_low_dim` suite (Lift, Can, Square, Transport, Tool Hang). The benchmark
contract is the
[robomimic protocol](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/ROBOMIMIC_EVALUATION_PROTOCOL.md).

| File | Purpose |
| --- | --- |
| `robomimic_policy.py` | The Policy: loads all five checkpoints with robomimic's `policy_from_checkpoint` and routes each episode to its task's model |
| `model_zoo.py` | The pinned checkpoint table: task, environment, file name, URL, size, SHA-256 |
| `download_model_zoo.py` | Downloads the five checkpoints sequentially and verifies their SHA-256 |
| `compare_native.py` | Runs the pinned upstream `run_trained_agent.py` for every seed of a Kit run and compares episode by episode |
| `roboscope-policy.json` | Policy process contract template |

## What the Policy keeps from upstream

The model-specific parts of robomimic's `run_trained_agent.py` stay in this Policy, not in the kit:

- observations go through robomimic's `RolloutPolicy` (tensor, batch, `float32` on the device, and
  observation normalization when a checkpoint carries statistics; the five PH low-dim checkpoints carry none);
- BC-RNN's LSTM hidden state and its every-`rnn.horizon`-steps (10) re-initialization counter are kept per
  episode and swapped in around each call, so one replica serves concurrent episodes without mixing them;
- the GMM head samples from the torch RNG. Each episode owns a torch RNG state seeded from its seed
  (`EpisodeContext.metadata["seed"]`), as upstream `torch.manual_seed(seed)` does before its loop;
- `reset` checks that the episode's environment name and horizon equal the checkpoint's `env_metadata` and
  `experiment.rollout.horizon`.

The Policy returns one 7-dimensional action (14 for Transport) per call, unclipped, exactly as upstream passes
it to `env.step`.

## Run

1. Build the simulator runtime; it also holds the pinned robomimic and torch that this Policy uses:

   ```bash
   benchmarks/robomimic/build-runtime.sh /sim/robomimic
   ```

2. Download the checkpoints (about 40 MB). The model zoo publishes no license of its own; check the
   robomimic terms before you use them:

   ```bash
   python examples/robomimic/download_model_zoo.py /models/robomimic-zoo
   ```

3. Copy `roboscope-policy.json` and replace the paths. The command runs the runtime's own Python with the
   kit source on `PYTHONPATH`, so the Policy uses exactly the robomimic and torch of the native comparison
   and nothing is installed into the runtime.

4. Evaluate with the benchmark CLI from any Python 3.11+ that has `roboscope-kit` installed:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m roboscope_kit.runtime.benchmark \
       --benchmark robomimic --suite ph_low_dim --quick-test \
       --checkpoint-source /models/robomimic-zoo \
       --benchmark-manifest /sim/robomimic/benchmark-manifest.json \
       --policy-config /path/to/roboscope-policy.json \
       --output-dir /runs/robomimic-quick
   ```

   Drop `--quick-test` for the full 5 x 50 evaluation; add `--concurrency K --replicas G` to skip autotuning.

5. Compare with upstream:

   ```bash
   python examples/robomimic/compare_native.py --runtime /sim/robomimic \
       --checkpoint-dir /models/robomimic-zoo --kit-run /runs/robomimic-full \
       --output /runs/robomimic-native --jobs 6
   ```
