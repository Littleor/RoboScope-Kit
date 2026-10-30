# RoboScope Kit

RoboScope Kit connects your model to RoboScope's simulation benchmarks. You write one `Policy` class in
Python; RoboScope runs the simulators, schedules the evaluation, and scores the results.

This repository is the single source of truth for RoboScope simulation evaluation. It contains:

- the `roboscope-kit` Python package, with the public `Policy` API and the platform benchmark runtime;
- the launchers and runtimes for the shared benchmark environments;
- engineering demos that validate the custom-image path.

Robot control, policy training, dataset collection, and generic LeRobot simulator code do not belong here.

## Install

```bash
python -m pip install roboscope-kit
```

```python
import roboscope_kit
```

Python 3.11 or newer is required. Start with the [package guide](packages/roboscope-kit/README.md), which
shows a complete `Policy`. The exact LIBERO contract (entrypoint, observations, actions, episodes, results)
is defined in the [LIBERO Evaluation Protocol](packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `packages/roboscope-kit/` | The `roboscope-kit` package: public `Policy` API, protocol client and server, platform benchmark runtime, and the evaluation protocol documents |
| `benchmarks/libero_official/` | LIBERO entrypoint (`/root/startbenchmark.sh`), runner, and self-test |
| `benchmarks/starvla_platform_runtime/` | Image-local Control Runtime installer, the per-benchmark `benchmark-image.json` manifests, and the thin launchers for CALVIN, LIBERO, LIBERO-Plus, MetaWorld, RoboCasa, and RoboTwin |
| `benchmarks/robotwin_official/` | RoboTwin launch and seed-cache tooling |
| `examples/custom_image_validation/` | Engineering demos for the custom-image path; they are not part of the product support matrix |
| `tests/async_inference/` | Autotune and LIBERO controller regressions |
| `ci/` | Build, compatibility, and PyPI release scripts used by GitHub Actions |

## Development

Install from source and run the checks:

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib
PYTHONPATH=src:../.. python -m pytest tests ../../tests/async_inference -q
python -m ruff check src tests ../../benchmarks/robotwin_official

cd ../..
benchmarks/libero_official/startbenchmark.sh --self-test
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

Real image installation, simulator episodes, and model loading need the documented target environment and
credentials.

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

## Recovering a lost shared LIBERO runtime

The shared disk keeps the LIBERO launch scripts and runtime archive under
`/root/autodl-fs/benchmarks/libero/`. Each evaluation extracts the archive into
`/root/benchmark-runtime/libero/`. The benchmark manifest always comes from the image's own Control Runtime,
so restoring shared files never changes an image's `roboscope-kit` version or benchmark configuration,
and a local manifest that differs from the image's copy is rejected.

If the whole shared LIBERO directory is lost, run the committed installer from a clean checkout on an
instance that has the source LIBERO environment under `/root`:

```bash
benchmarks/starvla_platform_runtime/install-platform-runtimes.sh \
  --recover-libero /root /root/autodl-fs/benchmarks "$(git rev-parse HEAD)"
```

It checks the source LIBERO environment, rebuilds the compressed runtime, records its SHA-256 and the
recovery commit, and publishes the archive and launch scripts together. It refuses to overwrite an existing
LIBERO directory. Run it under `tmux`, keep the recovery log, and verify a real
episode before treating the recovery as complete. Other benchmarks need their own runtime archives; this
command does not rebuild them.

## License

Copyright 2026 RoboScope Team. Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
