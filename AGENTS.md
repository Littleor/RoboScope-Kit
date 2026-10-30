# RoboScope Kit Agent Guide

## Scope

This repository owns RoboScope's `roboscope-kit` Python package, official evaluation images, benchmark adapters, evaluation launchers, and benchmark-runtime documentation. Do not put robot control, training, dataset collection, or generic LeRobot engine code here.

## Source of truth

- `packages/roboscope-kit/` owns the public Policy contract and platform runtime.
- `benchmarks/libero_official/` owns the validated official base image.
- `official_images/` owns model-specific official images.
- `benchmarks/starvla_platform_runtime/` owns the image-local Control Runtime installer and the thin launchers for shared Benchmark environments.
- `examples/custom_image_validation/` owns engineering demos used to validate the user-owned custom-image path. Demo models remain separate from the official image catalog and product support matrix.

Do not add compatibility wrappers for relocated paths. Update the source of truth and all callers directly.

## Shared runtime changes

Use committed runtime patch code as the exclusive write path for every runtime
on the shared disk, including runtimes under `/root/autodl-fs/benchmarks`. This
rule covers patches consumed by official images, custom personal images, and
every benchmark runtime. Express every runtime change in the owning, existing
runtime patch code in this repository, commit that code, and apply the change
by executing it. A runtime change is reproducible and complete only when a
clean shared-disk runtime plus the committed patch code reproduces the result.

## Official image single source of truth

A baked official image must be the byte-exact output of the committed image-build
scripts and manifests in this repository. Manual in-image edits are forbidden:
any change to a benchmark image (including per-benchmark
`benchmark-image.json` `defaults`) must land as a commit here and be applied by
re-baking with the documented scripts, never by hand-editing a file inside an
already-saved AutoDL image.

The per-benchmark `benchmark-image.json` files under
`benchmarks/starvla_platform_runtime/*/` are the single source of truth for what
runs inside a benchmark image. The bake pipeline ships them into the Control
Runtime as an immutable reference
(`/opt/roboscope/control-runtime/benchmark-manifests/<name>/benchmark-image.json`)
and checksums them into `runtime.sha256`; `run-benchmark.sh` materializes the
image's manifest under `/root/benchmark-runtime/<name>/benchmark-image.json`,
and compares an existing local manifest against that reference at evaluation
start. Any deviation aborts evaluation. Shared-disk installations do not select
the manifest or SDK version for a saved image.

## Documentation

Keep the SDK README, protocol documents, and image handoff documents consistent when user-facing evaluation behavior changes.

## Issue and PR titles

Use `type(scope): description` for new Issues and PRs, following the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) style.
Write Issue and PR titles and descriptions in English.

- type (lowercase): `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `build`,
  `ci`, or `chore`. Use `chore` for tracking Issues.
- scope (required, lowercase): a benchmark or component name, such as
  `robodojo`, `libero-pro`, `sdk`, or `autotune`; use `repo` for
  repository-wide work. Reuse existing scopes.
- description: a concise statement of the problem or intended outcome.

Examples:

- `feat(robodojo): integrate RoboDojo simulation evaluation with a correctness baseline`
- `perf(robodojo): improve RoboDojo evaluation throughput`

## Checks

### Python compatibility

The SDK's `pyproject.toml` → `tool.roboscope.python-support` table is the source
of truth for interpreter support. Maintained CPython versions follow the
official released bugfix/security lifecycle, with Python 3.11 as the minimum
installation and shared-source/API baseline. Python 3.8–3.10 are unsupported,
including pinned simulator and model environments that import the SDK.
Do not add legacy interpreter compatibility tiers or workarounds.

Keep distribution metadata, dependency resolution, Ruff and CI consistent. CI must build/install the sdist and wheel, import SDK modules and
run the existing regressions on every declared interpreter. Do not add tests;
change an existing test's compatibility syntax only after reproducing its failure.
Use `ci/python_support.py check` with Python 3.11+ to check policy
against the official release-cycle API.
GitHub Actions is the sole PyPI publisher, gated by its complete compatibility
matrix.

Every RoboScope Kit code change must bump the `roboscope-kit` version in both
`packages/roboscope-kit/pyproject.toml` and its `uv.lock`, then bake
the affected official image with the documented image-build scripts. A
code-only commit without a matching version and image bake is incomplete.

Run the existing SDK and moved evaluation tests after changes:

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib
PYTHONPATH=src:../.. python -m pytest tests ../../tests/async_inference -q
python -m ruff check src tests ../../benchmarks/robotwin_official
```

For LIBERO image changes, also run the hermetic self-test and build the commit-pinned bundle. Real image baking, simulator episodes, and model loading require the documented target environment.
