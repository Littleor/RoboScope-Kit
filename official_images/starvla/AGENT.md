# StarVLA Official Image

This directory owns the Benchmark-independent StarVLA official image.

Allowed content:

- pinned StarVLA runtime integration;
- RoboScope SDK Policy process;
- config-driven full-checkpoint loader;
- run-snapshot exporter;
- image build, install, verify, cleanup, and platform-selected Benchmark delegate scripts.

Do not add Benchmark names, suites, simulators, evaluators, workers, camera
layouts, action dimensions, gripper semantics, or scoring logic. Cloud selects
the exact platform runtime with `ROBOSCOPE_PLATFORM_ENTRYPOINT`; Benchmark code
belongs to the corresponding runtime under `/root/autodl-fs/benchmarks`. The
complete Control SDK and its dedicated Python are image-local under
`/opt/roboscope/control-runtime`; only simulator, dataset, and thin launcher
assets remain in AutoDL file storage.

The existing validated Benchmark-specific official image remains unchanged
until this new image passes real quick and formal evaluation. Retire the old
image and its directory in a separate change after production cutover.

SDK 0.7.0b58 and newer require Python 3.11 or newer in every SDK-importing
process. Verify the Policy interpreter and every selected Benchmark's simulator
interpreter before baking; the dedicated Control Python does not upgrade those
environments. Pinned Python 3.8–3.10 runtimes are unsupported and require separate
upgrade and acceptance before adopting this SDK release.
