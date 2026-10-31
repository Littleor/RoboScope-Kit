# RoboScope Kit Agent Guide

## Scope

This repository owns the `roboscope-kit` Python package (import name `roboscope_kit`), the benchmark
adapters and launchers, the shared benchmark runtimes, and their documentation. Do not put robot control,
training, dataset collection, or generic LeRobot engine code here.

A change is complete when it is committed, carries the required version bump, and is merged.

## Where things live

| Path | Owns |
| --- | --- |
| `packages/roboscope-kit/` | The public `Policy` API, protocol client and server, platform benchmark runtime, and the evaluation protocol documents |
| `benchmarks/libero_official/` | The LIBERO entrypoint (`startbenchmark.sh`), runner, and hermetic self-test |
| `benchmarks/starvla_platform_runtime/` | The Control Runtime installer, the per-benchmark `benchmark-image.json` manifests, and the thin launchers for shared benchmark environments |
| `benchmarks/robotwin_official/` | RoboTwin launch and seed-cache tooling |
| `examples/custom_image_validation/` | Engineering demos for the custom-image path; demo models are not part of the product support matrix |
| `tests/async_inference/` | Autotune and LIBERO controller regressions |
| `ci/` and `.github/workflows/` | Build, compatibility, and PyPI release automation |

When a path moves, update the source and every caller directly. Do not add compatibility wrappers or
aliases for old paths or names.

## Shared runtime changes

Every runtime on the shared disk, including those under `/root/autodl-fs/benchmarks`, is changed only by
committed runtime patch code. Put the change in the existing patch code that owns that runtime, commit it,
and apply it by running it. A runtime change is complete only when a clean shared-disk runtime plus the
committed patch code reproduces the result.

## Benchmark manifests

The `benchmark-image.json` files under `benchmarks/starvla_platform_runtime/*/` are the single source of
truth for what runs in each benchmark. The Control Runtime installer copies them to
`/opt/roboscope/control-runtime/benchmark-manifests/<name>/benchmark-image.json` and checksums them into
`runtime.sha256`. At evaluation start, `run-benchmark.sh` writes that manifest to
`/root/benchmark-runtime/<name>/benchmark-image.json` and aborts if an existing local manifest differs.
Shared-disk installations never choose the manifest or the `roboscope-kit` version. Change a manifest only
by committing it here.

## Versioning and releases

Every code change bumps the `roboscope-kit` version in both `packages/roboscope-kit/pyproject.toml` and
`packages/roboscope-kit/uv.lock`, and adds an entry to `packages/roboscope-kit/CHANGELOG.md`. A code change
without a version bump is incomplete. Documentation-only changes do not need a bump.

GitHub Actions is the only PyPI publisher. When a commit on `main` carries a version that is not on PyPI
yet, the `SDK compatibility` workflow builds the wheel and sdist, runs the full compatibility matrix,
uploads with the `PYPI_API_TOKEN` repository secret, and verifies the published hashes and a clean install.
Pull requests never publish. To retry a failed upload, run the workflow manually on `main`. Do not upload
to PyPI by hand.

## Documentation

Keep the package README, the protocol documents, and the handoff documents consistent whenever
user-facing evaluation behavior changes. The package README is also the PyPI project page, so link to
other repository files with absolute GitHub URLs.

## Issue and PR titles

Use `type(scope): description` for new Issues and PRs, following
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Write titles and descriptions in
English.

- type (lowercase): `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `build`, `ci`, or `chore`. Use
  `chore` for tracking Issues.
- scope (required, lowercase): a benchmark or component name, such as `robodojo`, `libero-pro`, `sdk`, or
  `autotune`; use `repo` for repository-wide work. Reuse existing scopes.
- description: a concise statement of the problem or intended outcome.

Examples:

- `feat(robodojo): integrate RoboDojo simulation evaluation with a correctness baseline`
- `perf(robodojo): improve RoboDojo evaluation throughput`

## Python compatibility

`tool.roboscope.python-support` in `packages/roboscope-kit/pyproject.toml` is the source of truth for
interpreter support. Supported versions follow the official CPython bugfix/security lifecycle, with
Python 3.11 as the minimum for installation and for the shared source and API baseline. Python 3.8–3.10
are unsupported, including pinned simulator and model environments that import the package. Do not add
legacy interpreter tiers or workarounds.

Keep the distribution metadata, dependency resolution, Ruff, and CI consistent. On every declared
interpreter, CI must build and install the sdist and wheel, import every module, and run the existing
regressions. Do not add tests; change an existing test's compatibility syntax only after reproducing its
failure. Check the policy against the official release-cycle API with `python ci/python_support.py check`
(Python 3.11+).

## Checks

Run after every change:

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib
PYTHONPATH=src:../.. python -m pytest tests ../../tests/async_inference -q
python -m ruff check src tests ../../benchmarks/robotwin_official
```

For LIBERO changes, also run the hermetic self-test:

```bash
benchmarks/libero_official/startbenchmark.sh --self-test
```

Real simulator episodes and model loading need the documented target environment.
