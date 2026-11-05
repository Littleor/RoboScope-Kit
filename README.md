<h1 align="center">RoboScope Kit</h1>

<p align="center">
  <strong>Unified interfaces for fast and scalable embodied-AI evaluation</strong>
</p>

<p align="center">
  <a href="https://mint-sjtu.github.io/"><img alt="MINT Lab@SJTU" src="https://img.shields.io/badge/MINT%20Lab%40SJTU-1565c0"/></a>
  <a href="https://arxiv.org/abs/2611.03850"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2611.03850-b31b1b.svg"/></a>
  <a href="https://pypi.org/project/roboscope-kit/"><img alt="PyPI" src="https://img.shields.io/pypi/v/roboscope-kit"/></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.03850">Paper</a> ·
  <a href="https://mint-sjtu.github.io/RoboScope-Kit/">Project page</a> ·
  <a href="packages/roboscope-kit/README.md">Package guide</a> ·
  <a href="packages/roboscope-kit/CHANGELOG.md">Changelog</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img src="docs/libero-race.gif" width="100%" alt="LIBERO-Long, 500 episodes on one RTX 4090: RoboScope Kit finishes in 8.1 minutes, 6.16× faster than the official StarVLA evaluation"/>
</p>

<p align="center">
  <img src="docs/robotwin-race.gif" width="534" alt="RoboTwin Easy, 500 episodes: from 359.7 minutes on one GPU to 14.4 minutes on four GPUs, 25.04× faster"/>
</p>

RoboScope Kit evaluates your model on standard robot simulation benchmarks. You wrap a model once in a Python
`Policy` class; the benchmark runtime drives the simulators, schedules the evaluation, and scores the results.
Each benchmark is wrapped once in an adapter around its own evaluator, so its rules and scoring stay unchanged.

## Highlights

- 🚀 **Faster evaluation:** on 500 episodes with a StarVLA checkpoint, evaluation is **6.16×**
  faster than the official StarVLA evaluator on LIBERO-Long (one RTX 4090, CUDA MPS) and **25.04×** faster on
  RoboTwin Easy (four RTX 4090s vs. a one-GPU baseline). See [Results](#results).
- 🧩 **One interface:** every supported benchmark runs through the same `Policy` API and sends its canonical
  observations unchanged.
- ⚙️ **Measured concurrency:** simulator Workers and Policy replicas run as separate pools. Autotune sizes both
  from real probe episodes, or you fix them with `--concurrency` and `--replicas`.
- 🌐 **Scale-out (Beta):** an evaluation can be sharded deterministically across GPU machines and merged
  afterwards.
- 📦 **Reproducible environments:** each benchmark's simulator is built from pinned sources by one script.
- ♻️ **Recoverable runs:** Work Items are claimed from a durable queue. A failed Worker's item returns to the
  queue, and a finished item is scored exactly once.

## Architecture

![Overview of the RoboScope Kit evaluation architecture](docs/overview.webp)

*The example metrics are for 500 LIBERO-Long episodes with 12 Workers and 3 replicas on one RTX 4090. See the
[interactive overview](https://mint-sjtu.github.io/RoboScope-Kit/#design).*

## Results

![Evaluation speedups on LIBERO-Long and RoboTwin Easy](docs/speedup.gif)

From the [technical report](https://arxiv.org/abs/2611.03850), each run covering 500 episodes with a StarVLA
checkpoint. Evaluation time excludes initialization, concurrency search or offline tuning, and cleanup.
LIBERO-Long, on one 48 GiB RTX 4090, with success rates equivalent to the official evaluation's:

| Configuration | vs. official StarVLA | vs. RLinf | vs. vla-eval |
| --- | ---: | ---: | ---: |
| 12 Workers, 3 replicas | **4.45×** | **2.78×** | **1.52×** |
| 16 Workers, 4 replicas, CUDA MPS | **6.16×** | **3.85×** | **2.10×** |

RoboTwin Easy, on four GPUs with chunk-boundary observation, takes 14.4 minutes instead of 359.7 with one
simulator and one model on one GPU (**25.04×**), with identical success outcomes and terminal steps. On the
same four GPUs, one simulator and one model per GPU takes 98.9 minutes, so RoboScope Kit is **6.88×** faster on
equal hardware.

## Supported benchmarks

| Benchmark | Suites | Contract |
| --- | --- | --- |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` | [LIBERO protocol](packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md) |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | [LIBERO protocol](packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md) |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | `demo_clean`, `demo_randomized` | [Seed protocol](benchmarks/robotwin/SEED_PROTOCOL.md) |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | `arx_x5`, `arx_x5_randomized` | [RoboDojo guide](benchmarks/robodojo/README.md) |
| [RoboCasa](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) | `tabletop` | [Manifest](benchmarks/robocasa/benchmark-manifest.json) |
| [CALVIN](https://github.com/mees/calvin) | `task_D_D`, `task_ABC_D`, `task_ABCD_D` | [CALVIN protocol](packages/roboscope-kit/CALVIN_EVALUATION_PROTOCOL.md) |
| MetaWorld | `mt50` | [MetaWorld protocol](packages/roboscope-kit/METAWORLD_EVALUATION_PROTOCOL.md) |

The simulators, datasets, and assets are not dependencies of the PyPI package; `build-runtime.sh` builds them
separately. Simulator builds, real episodes, and model loading need Linux x86_64, a GPU, and network access.

## Ready-made policies

Start from an existing `Policy` if you use one of these model families:

| Model | Benchmarks | Example |
| --- | --- | --- |
| [StarVLA](https://github.com/starVLA/starVLA) | LIBERO, LIBERO-Plus, RoboTwin, RoboCasa, CALVIN, MetaWorld, with a published checkpoint for each | [`examples/starvla/`](examples/starvla/README.md) |
| [X-VLA](https://github.com/2toINF/X-VLA) | LIBERO, LIBERO-Plus, RoboTwin, CALVIN | [`examples/xvla/`](examples/xvla/README.md) |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | RoboDojo | [`examples/xpolicylab/`](examples/xpolicylab/README.md) |

## Quick start

### 1. Install

Python 3.11 or newer is required.

```bash
python -m pip install roboscope-kit
```

### 2. Write a Policy

`load_model` stands for your own model code:

```python
import numpy as np

from roboscope_kit import ActionChunk, BenchmarkName, Policy, PolicyCapabilities, run


class UserPolicy(Policy):
    def load(self, context):
        self.model = load_model(context.checkpoint_path)
        return PolicyCapabilities(
            supported_benchmarks=(BenchmarkName.LIBERO,),
            required_features=("images.primary", "state.proprioception"),
            supported_suites={BenchmarkName.LIBERO: ("libero_10",)},
            supported_action_chunk_sizes=(1,),
        )

    def act(self, observation):
        return ActionChunk(np.asarray(self.model.predict(observation.features), dtype=np.float32))


if __name__ == "__main__":
    run(UserPolicy())
```

### 3. Tell the runtime how to start it

Save this as `roboscope-policy.json`:

```json
{
  "schema_version": 2,
  "command": ["/opt/my-policy/bin/python", "/opt/my-policy/user_policy.py"],
  "cwd": "/opt/my-policy"
}
```

### 4. Build the simulator and evaluate

From a checkout of this repository:

```bash
benchmarks/libero/build-runtime.sh /sim/libero

python -m roboscope_kit.runtime.benchmark \
    --benchmark libero --suite libero_10 --checkpoint-source /models/my-checkpoint \
    --benchmark-manifest /sim/libero/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/libero_10
```

Add `--quick-test` for a short real evaluation with one Worker and one Server. Results, progress, and logs are
written under `--output-dir`.
Use a new or empty output directory for each run, including `--self-test`. Only a directory containing
`errors.jsonl` alone can be retried in place; existing run artifacts are never resumed or overwritten.
Keep shell and scheduler logs outside the output directory.

The [package guide](packages/roboscope-kit/README.md) covers the rest: how evaluation works, automatic
tuning, writing a Policy, CUDA MPS, and cross-instance sharding.

## Repository layout

| Path | Contents |
| --- | --- |
| `packages/roboscope-kit/` | The `roboscope-kit` package: public `Policy` API, protocol client and server, benchmark runtime, and the evaluation protocol documents |
| `benchmarks/<benchmark>/` | For CALVIN, LIBERO, LIBERO-Plus, MetaWorld, RoboCasa, RoboDojo, and RoboTwin: the `benchmark-manifest.json` and `build-runtime.sh`, which builds the pinned simulator environment; RoboTwin also has its launcher and seed-cache tooling |
| `examples/starvla/` | The StarVLA `Policy` and one demo per benchmark, including the configurations behind the RoboScope Kit technical report |
| `examples/xvla/` | An X-VLA `Policy` example: a stateful Policy serving several benchmarks |
| `examples/xpolicylab/` | An XPolicyLab `Policy` example: serves any XPolicyLab policy on RoboDojo, validated with Simate-beta |
| `ci/` | Build, compatibility, and PyPI release scripts used by GitHub Actions |

## Contributing

Contributions are very welcome! 🎉 Bug reports, documentation fixes, new `Policy` examples, new benchmarks, and
performance work all help, and questions are contributions too. Open an
[Issue](https://github.com/MINT-SJTU/RoboScope-Kit/issues) to report a bug or propose a change, and read
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how to set up a checkout, run the checks, and what a pull request
needs. Performance pull requests need a same-machine A/B measurement.

## Citation

If you find RoboScope Kit helpful in your research, please consider citing our work:

```bibtex
@misc{wu2026roboscopekit,
  title         = {RoboScope Kit: Unified Interfaces for Fast and Scalable Embodied-AI Evaluation},
  author        = {Wu, Deng-Ke and Jin, Yifan and Yang, Yida and Qian, Lei and Zhou, Zhaoye and Zhang, Hongru and Ye, Kai and Hu, Shuyuan and Mao, Yiran and Ye, Zewei and Zhao, Bo},
  year          = {2026},
  eprint        = {2611.03850},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

## License

Copyright 2026 RoboScope Team. Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).

## Community

Ask questions, share your results, and discuss new benchmarks in
[GitHub Discussions](https://github.com/MINT-SJTU/RoboScope-Kit/discussions). Report bugs as
[Issues](https://github.com/MINT-SJTU/RoboScope-Kit/issues).

WeChat users can also scan the QR code to join the RoboScope Kit group (the invitation is valid before
November 12, 2026):

<img src="docs/community-qr.png" alt="WeChat QR code for the RoboScope Kit community" width="240">
