# Contributing to RoboScope Kit

Contributions are very welcome! 🎉 Bug reports, documentation fixes, new `Policy` examples, new benchmarks, and
performance work all help, and questions are contributions too. If you are unsure where to start, open an
[Issue](https://github.com/MINT-SJTU/RoboScope-Kit/issues) or ask in [GitHub Discussions](https://github.com/MINT-SJTU/RoboScope-Kit/discussions).

## How to contribute

- **Report a bug** with the command you ran, the `roboscope-kit` version, the benchmark and suite, what you
  expected, and the relevant logs from `--output-dir`.
- **Contribute a new benchmark.** New benchmarks are very welcome! Add its adapter under
  `packages/roboscope-kit/`, and its `benchmark-manifest.json` and `build-runtime.sh` under
  `benchmarks/<benchmark>/`, following the existing benchmarks. Keep its official rules and scoring unchanged,
  and show that results match its official evaluator.
- **Propose larger design changes first.** For a change to the runtime's design or the public API, open an
  Issue so we can agree on the approach before you invest the time.
- **Open a pull request** from a branch of your fork. Keep it to one problem, describe what changed, why, and
  how you checked it, and run the checks in [Development](#development) before you push.

[`AGENTS.md`](AGENTS.md) has the full repository rules. The ones contributors most often need:

- **Titles** follow `type(scope): description` ([Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)),
  in English. The type is one of `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `build`, `ci`, or
  `chore`; the scope is a benchmark or component, such as `robodojo`, `libero`, `sdk`, or `autotune`, or
  `repo` for repository-wide work. For example: `perf(robodojo): improve RoboDojo evaluation throughput`.
- **Code changes bump the version** in `pyproject.toml` and `uv.lock` and add a `CHANGELOG.md` entry, as
  described in [Releases](#releases). Documentation-only changes need no bump.
- **Simulator environments change only through `benchmarks/<benchmark>/build-runtime.sh`** and its pin
  files, so a clean build reproduces them. Do not patch an installed environment by hand.
- **Benchmark rules and scoring stay unchanged.** Keep each manifest consistent with its build script, and do
  not change frozen protocol values (such as the RoboTwin seed protocol) without regenerating what depends on
  them.
- **Keep the documentation consistent** when user-facing evaluation behavior changes: the top-level
  README, the package README, the protocol documents, and the example READMEs.

## Coding agents are welcome

You may use coding agents and AI assistants (Claude Code, Codex) to write code,
tests, or documentation; point them to [`AGENTS.md`](AGENTS.md) for the repository rules. You remain the
author of what you submit: read and understand the whole change, run the checks yourself, be ready to explain
it in review, and make sure every number and test result in the description comes from a run that actually
happened.

## Performance changes need experimental evidence

Speed is a core promise of RoboScope Kit, and performance work is especially welcome. Because performance is
easy to mismeasure, a `perf` pull request, or any change claiming faster or lighter evaluation, must include
experimental evidence; a claim without measurements cannot be merged. Please provide:

1. **A same-machine A/B comparison:** the baseline commit and your change, run back to back on the same
   machine with the same benchmark, suite, checkpoint, episode count, and Worker, replica, and CUDA MPS
   settings. Numbers from different machines or days are not comparable.
2. **The setup and exact commands:** CPU, GPU model and count, driver and CUDA version, and the commit of
   each side, so a reviewer can reproduce the runs.
3. **The results:** evaluation wall-clock time (excluding initialization, concurrency search, and cleanup,
   unless the change targets them), steps/s or episodes/s, and any resource metric you claim to improve.
   Repeat each side when you can (three runs is a good default) and report the median and spread.
4. **Evidence that the evaluation is unchanged:** success outcomes that match the baseline (per-episode
   outcomes and terminal steps where the benchmark is deterministic, otherwise equivalent success rates),
   with every episode finished exactly once and no missing or duplicate records.
5. **The raw data:** the run summaries and logs of both sides, attached or linked.

Without the hardware for a full run, open a draft pull request with what you could measure (a `--quick-test`
run or a profile, say) and tell us; we are glad to help. Negative results are useful too: an attempt that did
not help, with its measurements, saves others from repeating it.


## Development

Install from source and run the checks:

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib
PYTHONPATH=src:../.. python -m pytest tests -q
python -m ruff check src tests ../../benchmarks/robotwin
```

To build a benchmark's simulator environment and check it against the kit:

```bash
benchmarks/libero/build-runtime.sh /sim/libero
SELF_TEST_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/libero-self-test.XXXXXX")
python -m roboscope_kit.runtime.benchmark --benchmark libero --self-test \
    --benchmark-manifest /sim/libero/benchmark-manifest.json --output-dir "$SELF_TEST_OUTPUT"
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

By contributing, you agree that your contributions are licensed under the project's
[Apache License 2.0](LICENSE).
