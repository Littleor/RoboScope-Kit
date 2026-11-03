# RoboScope Kit

RoboScope Kit evaluates your model on standard robot simulation benchmarks. You write one `Policy` class in
Python; the benchmark runtime drives the simulators, schedules the evaluation, and scores the results.

This repository contains:

- the `roboscope-kit` Python package, with the public `Policy` API and the benchmark runtime;
- one directory per benchmark, with its manifest and the script that builds its simulator environment;
- example Policies for StarVLA and X-VLA.

Robot control, policy training, dataset collection, and generic LeRobot simulator code do not belong here.

## Install

```bash
python -m pip install roboscope-kit
```

```python
import roboscope_kit
```

Python 3.11 or newer is required. Start with the [package guide](packages/roboscope-kit/README.md), which
shows a complete `Policy` and the command that evaluates it. The exact LIBERO contract (invocation,
observations, actions, episodes, results) is defined in the
[LIBERO Evaluation Protocol](packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md).

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

Simulator builds, real episodes, and model loading need Linux x86_64, a GPU, and network access.

## Releases

Releases are automatic. GitHub Actions (`.github/workflows/sdk-compatibility.yml`) is the only publisher.

1. Bump the version in `packages/roboscope-kit/pyproject.toml` and `uv.lock`, and add a `CHANGELOG.md`
   entry.
2. Merge the change to `main`.
3. CI builds the wheel and sdist once, tests them on every supported Python version, uploads them to PyPI
   with the `PYPI_API_TOKEN` repository secret, then checks the published hashes and a clean install.

A version that is already on PyPI is never uploaded again; CI only re-checks the install. Pull requests run
the tests but never publish. To retry a failed upload, run the `SDK compatibility` workflow manually on
`main`.

## License

Copyright 2026 RoboScope Team. Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
