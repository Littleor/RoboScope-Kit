# RoboScope Kit

RoboScope Kit is the source of truth for RoboScope simulation evaluation. It contains the public `roboscope-kit` Python package (Policy inference API), official evaluation-image build inputs, platform benchmark runtimes, and their validation assets.

## Repository layout

- `packages/roboscope-kit/`: public `Policy` API and platform-owned benchmark runtime.
- `benchmarks/libero_official/`: validated official base-image bundle, installer, verifier, and external `/root/startbenchmark.sh` contract.
- `official_images/starvla/`: benchmark-independent StarVLA official-image bundle.
- `benchmarks/starvla_platform_runtime/`: image-local Control Runtime installer plus shared launchers for CALVIN, LIBERO-Plus, RoboCasa, and RoboTwin.
- `benchmarks/robotwin_official/`: RoboTwin launch and seed-cache tooling.
- `benchmarks/`: official benchmark launchers, image-owned runtimes, and protocol assets.
- `examples/custom_image_validation/`: engineering demos that validate the user-owned custom-image path; these demos carry engineering evidence rather than product support status.

LeRobot's robot, policy-training, dataset, and generic simulator code is not owned here.

## Python package

Install `roboscope-kit` from source:

```bash
python -m pip install ./packages/roboscope-kit
```

Start with the [SDK guide](packages/roboscope-kit/README.md). The exact LIBERO entrypoint, observation, action, episode, and result contract is defined in [LIBERO Evaluation Protocol](packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md).

The SDK maintains released CPython versions in the official bugfix/security
lifecycle, with Python 3.11 as the minimum supported version. Wheels and source
distributions require Python 3.11 or newer; Python 3.8–3.10 are unsupported.
The package's `tool.roboscope.python-support` table owns the CI matrix;
GitHub pull requests check installed artifacts and run the existing regressions on every declared interpreter.

## Official base image

Build a commit-pinned bundle from a clean checkout:

```bash
benchmarks/libero_official/build_official_image_bundle.sh --ref origin/main
```

The bundle is written to `dist/official-image/`. Copy it into a fresh instance created from the current official image and run its `install.sh`; the installer verifies payload checksums, installs the image-local Control Runtime and SDK, runs the hermetic self-test, and clears user state before baking.

Building or pushing this repository does not publish an AutoDL image or change the Cloud image UUID. Follow [the official base image handoff](benchmarks/libero_official/AGENT.md) for the complete bake and rollout procedure.

### Recovering a missing shared LIBERO runtime

Runtime compression keeps the launch scripts and archives on the shared disk;
`/root/autodl-fs/benchmarks/libero/` must still exist. Each evaluation extracts
the archive under `/root/benchmark-runtime/libero/`. The launcher uses the
immutable benchmark manifest from that instance's image-local Control Runtime,
so restoring shared files does not replace a saved image's SDK or benchmark
configuration. An existing local manifest which differs from the image
reference is rejected.

If the entire shared LIBERO directory is lost, create a fresh instance from
the validated official base image and execute the committed installer from a
clean RoboScope Kit checkout:

```bash
benchmarks/starvla_platform_runtime/install-platform-runtimes.sh \
  --recover-libero /root /root/autodl-fs/benchmarks "$(git rev-parse HEAD)"
```

This checks the source LIBERO environment, rebuilds the compressed runtime,
records its SHA-256 and the recovery commit, and publishes the archive and
launch scripts together. It refuses to overwrite an existing LIBERO directory.
Run it under `tmux`, retain the source image ID and recovery log, and verify a
real episode before considering recovery complete. Other benchmarks require
their own runtime archives; this command does not reconstruct them.

## Validation

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib
PYTHONPATH=src:../.. python -m pytest tests ../../tests/async_inference -q
python -m ruff check src tests ../../benchmarks/robotwin_official

cd ../..
benchmarks/libero_official/startbenchmark.sh --self-test
```

Image installation and real simulator/model checks require the documented target runtimes and credentials.

To run the same installed-artifact compatibility gate as CI in a dedicated
environment for any declared interpreter:

```bash
python -m pip install build twine
bash ci/build-sdk.sh
bash ci/check-sdk.sh
```

The policy/lint job uses Python 3.11+ and checks the official CPython lifecycle,
package classifiers and Python 3.11 source/API baseline. Lifecycle
changes require updating the package policy and CI together.

## Releases

GitHub Actions is the sole SDK publisher. Pushes to GitHub `main` build the
wheel and sdist once, record their SHA-256 hashes, and require the complete
compatibility gate before publishing a version that PyPI does not have yet.
The publish job uses the repository secret `PYPI_API_TOKEN`, serializes releases,
checks the downloaded build hashes, uploads those exact artifacts, checks the
hashes exposed by PyPI, and installs the published wheel in a clean environment.
Runs with an existing version skip upload and artifact comparison, but still
verify installation from official PyPI. Pull requests and manual workflow runs
only check compatibility and never publish.

Set `PYPI_API_TOKEN` as a GitHub Actions repository secret before enabling
releases. Bump the SDK version and merge the checked PR to
GitHub `main` to release. This workflow does not bake or publish AutoDL images.
