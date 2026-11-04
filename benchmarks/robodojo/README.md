# RoboDojo simulation

`build-runtime.sh <install-dir>` builds the pinned Linux x86-64 simulator, downloads the benchmark assets,
and writes `benchmark-manifest.json` under the install directory. It requires `git`, `curl`, `sha256sum`,
the Debian system packages `cmake`, `build-essential`, and `ffmpeg`, a working CUDA build toolchain, and a
compatible NVIDIA GPU. The builder installs an isolated Miniforge, runs RoboDojo's `scripts/install.sh -i`,
then applies the Kit patches and checks the resulting environment. The upstream installer activates
`$HOME/miniconda3` when present, so the isolated build requires an account without that path. The asset
download is approximately 41 GB. Isaac Sim is licensed by NVIDIA; read its
[license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
and set `OMNI_KIT_ACCEPT_EULA=YES` yourself before building or running it. The build does not set
`PRIVACY_CONSENT`.

The `arx_x5` suite contains the 42 base simulation tasks at the pinned RoboDojo revision.
`arx_x5_randomized` contains the 12 additional randomized variants. The manifest records each task's
upstream default of 25 or 50 layouts and uses seed 1. `--episodes` can request a smaller common prefix;
requests above an upstream task limit are rejected. Work is divided into non-overlapping fixed-layout
shards. A shard missing a layout or returning incomplete results fails rather than reporting a partial run.

The Policy receives `images.head`, `images.left_wrist`, and `images.right_wrist` as RGB uint8 arrays,
`language.instruction` as a string, and `state.joints` as 14 float32 values ordered as six left-arm joints,
left gripper, six right-arm joints, right gripper. `state.ee_pose` contains both end-effector poses as
`[x, y, z, qw, qx, qy, qz]` in the environment frame. The default `robodojo.x5-joint-v1` action uses the
joint-state order for absolute targets and gripper commands. The `robodojo.x5-ee-compact-quat-v1` action
instead has `[x, y, z, qx, qy, qz, gripper]` for each arm; the missing quaternion `w` is reconstructed as
nonnegative. A Policy that needs every intermediate observation can declare
`step_observation_updates=("robodojo",)` and implement `observe(Observation)`: `act()` receives the
first observation of a chunk, and `observe()` receives the intermediate observations in step order.

The manifest also exposes the experimental `async_video`, `unpaced_render`, and `control_state_reuse`
switches, plus `physics_threads` (1, 4 or 8). They are disabled by default. `observation_queue_size` defaults
to 0 (synchronous) and can be set to 1–10 for bounded asynchronous observation delivery when the Policy
declares step updates. Pending updates finish before the next `act()` or episode end. These candidates
originate in @QLL-CCDD's RoboDojo work and should be evaluated for correctness and throughput for the
intended workload.

The simulator source, IsaacLab, CuRobo, XPolicyLab, package constraints, and the asset dataset are pinned
by the build script. Its submodule selections keep the upstream installer's `--remote` updates at those
tested revisions.
Model code and checkpoints are supplied separately through the standard Kit Policy configuration;
[`examples/xpolicylab/`](../../examples/xpolicylab/README.md) serves XPolicyLab policies this way.
The layout-sharding patch is adapted from @QLL-CCDD's RoboDojo work.

The asset downloader verifies the pinned inventory and each file's hash. `HF_ENDPOINT` selects the metadata
and primary download host; `ROBODOJO_ASSET_TRANSPORT=modelscope` uses ModelScope as a first byte source
with the same pinned hashes. `ROBODOJO_ASSET_INVENTORY` may point to pre-fetched inventory metadata when
the metadata service is unavailable, and `ROBODOJO_ASSET_WORKERS` controls download concurrency.
