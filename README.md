# RoboScope Kit

[![MINT Lab@SJTU](https://img.shields.io/badge/MINT%20Lab%40SJTU-1565c0)](https://mint-sjtu.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.03850-b31b1b.svg)](https://arxiv.org/abs/2611.03850)
[![PyPI](https://img.shields.io/pypi/v/roboscope-kit)](https://pypi.org/project/roboscope-kit/)

RoboScope Kit evaluates your model on standard robot simulation benchmarks, quickly and at scale. You wrap a
model once in a Python `Policy` class; the benchmark runtime drives the simulators, schedules the evaluation,
and scores the results. Each benchmark is wrapped once in an adapter around its own evaluator, so its rules
and scoring stay unchanged.

[Paper](https://arxiv.org/abs/2611.03850) · [Project page](https://roboscope.pages.dev/) ·
[Package guide](packages/roboscope-kit/README.md) · [Changelog](packages/roboscope-kit/CHANGELOG.md)

This repository contains:

- the `roboscope-kit` Python package, with the public `Policy` API and the benchmark runtime;
- one directory per benchmark, with its manifest and the script that builds its simulator environment;
- example Policies for StarVLA and X-VLA.

## Why RoboScope Kit

- **One interface.** Every supported benchmark runs through the same `Policy` API and sends its canonical
  observations unchanged.
- **Measured concurrency.** Simulator Workers and Policy replicas run as separate pools. Autotune sizes both
  from real probe episodes, or you fix them with `--concurrency` and `--replicas`.
- **Scale-out (Beta).** An evaluation can be sharded deterministically across GPU machines and merged
  afterwards.
- **Reproducible environments.** Each benchmark's simulator is built from pinned sources by one script.
- **Recoverable runs.** Work Items are claimed from a durable queue: a failed Worker's item returns to the
  queue, and a finished item is scored exactly once.

Results from the [technical report](https://arxiv.org/abs/2611.03850), each over 500 episodes with a StarVLA
checkpoint:

| Benchmark | Hardware | Configuration | Speedup |
| --- | --- | --- | --- |
| LIBERO-Long | One 48 GiB RTX 4090 | 12 Workers, 3 replicas | 4.45× over the official StarVLA evaluation, 2.78× over RLinf, 1.52× over vla-eval |
| LIBERO-Long | One 48 GiB RTX 4090 | 16 Workers, 4 replicas, CUDA MPS | 6.16× over the official StarVLA evaluation, 3.85× over RLinf, 2.10× over vla-eval |
| RoboTwin Easy | Four GPUs | Chunk-boundary observation | 25.04× over one simulator and one model on one GPU (359.7 min down to 14.4 min) |

LIBERO-Long success rates are equivalent to the official evaluation's; RoboTwin Easy has identical success
outcomes and terminal steps.

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

## Quick start

Python 3.11 or newer is required.

```bash
python -m pip install roboscope-kit
```

Write a `Policy`. `load_model` stands for your own model code:

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


run(UserPolicy())
```

Tell the runtime how to start it, in a `roboscope-policy.json`:

```json
{
  "schema_version": 2,
  "command": ["/opt/my-policy/bin/python", "/opt/my-policy/user_policy.py"],
  "cwd": "/opt/my-policy"
}
```

Build the simulator environment from a checkout of this repository, then evaluate:

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

The [package guide](packages/roboscope-kit/README.md) covers the rest: how evaluation works, automatic
tuning, writing a Policy, CUDA MPS, and cross-instance sharding.

## Repository layout

| Path | Contents |
| --- | --- |
| `packages/roboscope-kit/` | The `roboscope-kit` package: public `Policy` API, protocol client and server, benchmark runtime, and the evaluation protocol documents |
| `benchmarks/<benchmark>/` | For CALVIN, LIBERO, LIBERO-Plus, MetaWorld, RoboCasa, RoboDojo, and RoboTwin: the `benchmark-manifest.json` and `build-runtime.sh`, which builds the pinned simulator environment; RoboTwin also has its launcher and seed-cache tooling |
| `examples/starvla/` | The StarVLA `Policy` and one demo per benchmark, including the configurations behind the RoboScope Kit technical report |
| `examples/xvla/` | An X-VLA `Policy` example: a stateful Policy serving several benchmarks |
| `tests/async_inference/` | Autotune regressions |
| `ci/` | Build, compatibility, and PyPI release scripts used by GitHub Actions |

## Development

Install from source and run the checks:

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib
PYTHONPATH=src:../.. python -m pytest tests ../../tests/async_inference -q
python -m ruff check src tests ../../benchmarks/robotwin
```

To build a benchmark's simulator environment and check it against the kit:

```bash
benchmarks/libero/build-runtime.sh /sim/libero
python -m roboscope_kit.runtime.benchmark --benchmark libero --self-test \
    --benchmark-manifest /sim/libero/benchmark-manifest.json --output-dir /tmp/libero-self-test
```

To reproduce the CI compatibility check on the built wheel and sdist, run these with the interpreter you
want to test:

```bash
python -m pip install build twine
bash ci/build-sdk.sh
bash ci/check-sdk.sh
```

Supported Python versions follow the official CPython release lifecycle, with Python 3.11 as the minimum.
The list lives in `packages/roboscope-kit/pyproject.toml` under `tool.roboscope.python-support`; CI reads it
to build its test matrix and fails when it drifts from the official lifecycle.

## Releases

Releases are automatic. GitHub Actions (`.github/workflows/sdk-compatibility.yml`) is the only publisher.

1. Bump the version in `packages/roboscope-kit/pyproject.toml` and `uv.lock`, and add a `CHANGELOG.md`
   entry headed `## <version> - <YYYY-MM-DD>`.
2. Merge the change to `main`.
3. CI builds the wheel and sdist once, tests them on every supported Python version, uploads them to PyPI
   with the `PYPI_API_TOKEN` repository secret, then checks the published hashes and a clean install.

A version that is already on PyPI is never uploaded again; CI only re-checks the install. Pull requests run
the tests but never publish. To retry a failed upload, run the `SDK compatibility` workflow manually on
`main`.

## Citation

If you use RoboScope Kit in your research, please cite:

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
