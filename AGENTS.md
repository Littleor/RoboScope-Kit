# RoboScope Kit Agent Guide

## Scope

This repository owns the `roboscope-kit` Python package (import name `roboscope_kit`), the benchmark
adapters, the benchmark manifests and simulator build scripts, and their documentation. Do not put robot control,
training, dataset collection, or generic LeRobot engine code here.

A change is complete when it is committed, carries the required version bump, and is merged.

## Where things live

| Path | Owns |
| --- | --- |
| `packages/roboscope-kit/` | The public `Policy` API, protocol client and server, benchmark runtime, and the evaluation protocol documents |
| `benchmarks/<benchmark>/` | The benchmark's `benchmark-manifest.json`, the `build-runtime.sh` that builds its pinned simulator environment, and its pin files; RoboTwin also has its launcher and seed-cache tooling |
| `examples/starvla/` | The StarVLA `Policy` and one demo per benchmark, including the configurations behind the RoboScope Kit technical report |
| `examples/xvla/` | The X-VLA `Policy` example: a stateful Policy serving several benchmarks |
| `examples/xpolicylab/` | An XPolicyLab `Policy` example: serves any XPolicyLab policy on RoboDojo, validated with Simate-beta |
| `ci/` and `.github/workflows/` | Build, compatibility, and PyPI release automation |

When a path moves, update the source and every caller directly. Do not add compatibility wrappers or
aliases for old paths or names.

## Simulator environments

`benchmarks/<benchmark>/build-runtime.sh <install-dir>` is the only way a simulator environment is built or
changed. Put every source pin, package pin, checksum, and source patch in that script or its pin files, so a
clean build from the committed script reproduces the environment. Do not patch an installed environment by
hand.

## Benchmark manifests

`benchmarks/<benchmark>/benchmark-manifest.json` defines what runs in each benchmark: its suites, protocol,
runtime paths, and evaluation defaults. Runtime paths are relative to the manifest's directory; the build
script copies the manifest into `<install-dir>`, where those paths resolve. Keep the manifest and its build
script consistent, and never change frozen protocol values (such as the RoboTwin seed protocol and
generation digests) without regenerating what depends on them.

## Versioning and releases

Every code change bumps the `roboscope-kit` version in both `packages/roboscope-kit/pyproject.toml` and
`packages/roboscope-kit/uv.lock`, and adds an entry to `packages/roboscope-kit/CHANGELOG.md`. Head the entry
`## <version> - <YYYY-MM-DD>` with the merge date, never `Unreleased`: merging the version publishes it. A code
change without a version bump is incomplete. Documentation-only changes do not need a bump.

GitHub Actions is the only PyPI publisher. When a commit on `main` carries a version that is not on PyPI
yet, the `SDK compatibility` workflow builds the wheel and sdist, runs the full compatibility matrix,
uploads with the `PYPI_API_TOKEN` repository secret, and verifies the published hashes and a clean install.
Pull requests never publish. To retry a failed upload, run the workflow manually on `main`. Do not upload
to PyPI by hand.

## Documentation

Write everything in the repository in English: code, comments, documentation, and commit messages.

Keep the package README, the protocol documents, and the example READMEs consistent whenever
user-facing evaluation behavior changes. After a `--profile` run, add what it found (date, machine,
configuration, Kit version) to the benchmark's own documentation and link it from the Findings list in
`packages/roboscope-kit/docs/profiling.md`; update that file's coverage table when a benchmark is first
profiled. The package README is also the PyPI project page, so link to
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
PYTHONPATH=src:../.. python -m pytest tests -q
python -m ruff check src tests ../../benchmarks/robotwin
```

For a build script change, run `bash -n` and `shellcheck` on it. Real builds, simulator episodes, and model
loading need Linux x86_64, a GPU, and network access; after a build, check it with the kit's
`--self-test --benchmark-manifest <install-dir>/benchmark-manifest.json`.
