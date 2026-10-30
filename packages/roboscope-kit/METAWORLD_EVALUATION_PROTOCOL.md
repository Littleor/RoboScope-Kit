# MetaWorld MT50 Evaluation Protocol

This document defines the SDK-native RoboScope MetaWorld evaluation contract. The platform owns benchmark
enumeration, simulator processes, concurrency, recovery, and result aggregation. User inference code owns
only the public `Policy` lifecycle and model-specific preprocessing and postprocessing.

SDK 0.7.0b58 and newer require Python 3.11 or newer in every process that
imports the SDK, including Policy and simulator workers. Python 3.8–3.10
runtimes are unsupported; existing images require separate upgrade and
acceptance before adopting this release. See the [SDK Python policy](README.md#python-compatibility).

## 1. Benchmark scope

The only supported suite is `mt50`. The `benchmark` identifier is `metaworld` and the protocol version is
`metaworld-v1` (`metaworld.delta-eef-v1` action representation). `mt10`, `mt25`, `ml1`, `ml10`, `ml25`,
`ml45`, and the paper-leaderboard `easy` / `medium` / `hard` / `very_hard` groups are not submittable
suites. A formal pass evaluates all 50 Farama `ALL_V3_ENVIRONMENTS` tasks, one episode per training goal,
50 goals per task, 2,500 episodes in total, with an episode horizon of 500 environment steps.

The task slugs, their order, and the action-instruction text are frozen in the SDK. Instructions are the
verbatim v3.1.1 `docs/benchmark/task_descriptions.md` action sentences (original casing, punctuation, and
typos preserved, for example `disassemble-v3` = `Pick a nut out of the a peg.` and `push-back-v3` = the
`Pull` entry `Pull a puck to a goal.`). The `Randomize ...` environment notes are not instructions. The
evaluator worker asserts that the instruction keys equal `metaworld.env_dict.ALL_V3_ENVIRONMENTS` at
startup.

## 2. User policy lifecycle

The user entrypoint installs `roboscope-kit`, implements `Policy`, and ends with
`run(UserPolicy())`. RoboScope starts one process per selected replica and injects a Unix socket through
`ROBOSCOPE_INFERENCE_SOCKET`. The platform then calls:

1. `Policy.load(SessionContext)` once per replica with the resolved MetaWorld `BenchmarkSpec`;
2. `Policy.reset(EpisodeContext)` before every evaluated episode;
3. `Policy.act(Observation)` whenever the evaluator needs a new action chunk;
4. `Policy.end_episode(episode_id)` when the episode ends or the worker shuts down;
5. `Policy.unload()` when evaluation ends.

The worker keeps one persistent policy connection for the whole worker process; it never opens a temporary
connection per reset. `PolicyCapabilities` must declare MetaWorld, the `mt50` suite, every observation
feature the model consumes, and the supported action-chunk sizes. The observation specification is built
from the resolved evaluation configuration, so the features seen by `Policy.load` and by the evaluator
worker are always identical.

## 3. Canonical observation

Every `Observation` contains:

| Feature | Type | Shape | Construction |
| --- | --- | --- | --- |
| configured image feature | contiguous `uint8`, RGB/HWC | `output_height x output_width x 3` | one rendered camera per configured entry |
| `state.proprioception` | contiguous `float32` | `(4,)` | raw `observation[:4]` (end-effector xyz and normalized gripper aperture) |
| `language.instruction` | `str` | scalar | verbatim official instruction for the current task |

`episode_id` identifies the task and episode; `step` starts at zero for each episode and increases once per
environment `step()`, not once per policy request. Multi-camera images are rendered from the same
environment state within one observation; the worker never steps or resets the simulator between cameras,
and never duplicates one camera image to fake another view. The platform does not resize, normalize, or
transpose images for a model beyond the declared configuration; any further model-specific preprocessing
belongs inside the user policy.

### Camera configuration

The evaluation configuration has one JSON shape, shared by the Policy file declaration, the Cloud request,
and the runtime file the platform writes for the workers:

```json
{"cameras": [{"feature": "images.primary", "camera_name": "corner2", "render_width": 480,
  "render_height": 480, "camera_position": null, "rotation_deg": 180, "center_crop_ratio": 1.0,
  "output_width": 480, "output_height": 480}]}
```

Rules:

- 1 to 4 cameras. `feature` is required and must be `images.primary`, `images.secondary`,
  `images.tertiary`, `images.quaternary` in order and unique. `camera_name` is required and must be
  unique. An empty `cameras` list and `{}` are errors, and unknown fields are rejected.
- The platform default is the single camera above: `images.primary`, `corner2`, rendered and emitted at
  480x480, `camera_position = null`, `rotation_deg = 180`, `center_crop_ratio = 1.0`.
- `render_width`/`render_height`/`output_width`/`output_height` are integers in `[64, 1024]`.
  `rotation_deg` is counter-clockwise and one of `0`, `90`, `180`, `270`. `center_crop_ratio` is a finite
  number in `(0, 1]`. `camera_position` is `null` or a finite `[x, y, z]` in meters in the camera body's
  local frame.
- Per camera the platform renders, rotates counter-clockwise, center-crops (crop size is the rotated size
  times the ratio, floored; the top-left corner is half of the remaining size, floored; a zero-sized crop
  is rejected), resizes with the pinned Pillow bilinear filter when the size differs, and emits a
  contiguous RGB/HWC `uint8` array. The applied image shape is the configured `output_height x
  output_width x 3`.
- Cameras are resolved by MuJoCo camera name. A missing name is a hard error. A `camera_position` override
  is only allowed for a fixed-mode camera (including a fixed camera attached to a moving body); other
  camera modes are rejected instead of retargeted, and only the position changes while the upstream
  orientation is preserved. The configured camera names must exist in all 50 task models; the worker
  checks this at startup. The worker verifies the OpenGL framebuffer can hold the largest configured render
  size and releases every renderer at shutdown.

Proprioception stays a fixed 4-dimensional vector and the language instruction stays the official text.
Scoring rules, the task set, the goals, the 500-step horizon, and the difficulty-macro aggregation are not
user-configurable.

## 4. Action contract

`Policy.act()` returns an `ActionChunk` containing either one 4-dimensional action or a two-dimensional
array of 4-dimensional actions. The values are the native MetaWorld delta end-effector action:

```text
[dx, dy, dz, gripper]
```

`dx, dy, dz` are metric end-effector displacements and `gripper` is the gripper command, all in `[-1, 1]`
(`metaworld.delta-eef-v1`, `float32`). Actions from a chunk are executed in order, one environment step at
a time. A new observation is requested only when the current chunk is exhausted; the remaining chunk is
discarded when the episode succeeds or reaches 500 steps.

## 5. Goals, seeding, and the official rollout

The platform generates the complete `MT50(seed=42).train_tasks` artifact once per worker process, groups it
by `env_name`, and requires exactly 50 tasks x 50 goals. It records the canonical SHA-256 over
`env_name | NUL | task.data | NUL` in `MT50.train_tasks` order and asserts it against the frozen
`FROZEN_MT50_TASKS_SHA256`; a mismatch aborts the worker. The platform never builds 50 independent
`MT1(slug, seed=42)` environments, because `_make_tasks` consumes the generator across the whole task set
and an independent `MT1` slice is not the MT50 goal set.

The Farama evaluation function does not select goals, and the default environment factory samples tasks
with replacement. To hit every training goal exactly once, the platform builds its environment through the
pinned `metaworld._init_each_env(..., task_select="pseudorandom", use_one_hot=False,
max_episode_steps=500, render_mode="rgb_array", ...)` and enables
`toggle_sample_tasks_on_reset(True)`. The `PseudoRandomTaskSelectWrapper` advances to the next goal on
every reset and shuffles the list when it wraps; that first shuffle means a fresh worker starts with a
shuffled goal order. The worker seeds the environment RNG while constructing the environment, before any
reset or sample, because the task wrapper chooses the task before the inner `reset(seed=...)` runs. Goal
identity is the frozen-artifact index, not the shuffled position; each finished episode records its stable
`goal_id` and the SHA-256 of its frozen `Task`.

`seed = 42` is a platform reproduction parameter, not an upstream or paper standard. The platform does not
claim that an arbitrary model's training set is identical to this artifact without per-protocol training
artifact evidence.

## 6. Official rollout behavior

The worker calls the pinned upstream `metaworld.evaluation.evaluation(agent, eval_envs, num_episodes=N)`
and does not reimplement the rollout or success rule. An episode succeeds if `info["success"] == 1` at any
step; the reused `AutoTerminateOnSuccessWrapper` terminates the episode on success, and
`gym.wrappers.TimeLimit` truncates at 500 steps. A formal task evaluation uses `num_episodes=50`; the Quick
Test uses 1.

The single-task `gymnasium.vector.SyncVectorEnv` uses `AutoresetMode.SAME_STEP`. A thin
`RecordingVectorWrapper` records only the episode that just finished: it reads
`infos["final_info"]["success"][i]` and `infos["final_info"]["episode"]["l"][i]` for each finished
environment and fails loudly on missing fields. The final `done` triggers an automatic 51st reset; that
reset is not a 51st result, is never counted, and never receives the previous episode's goal.

Unexpected simulator, protocol, or policy errors terminate the worker and surface through the generic
runtime. They are not converted into failed tasks.

## 7. Aggregation

Each finished episode contributes one record with a stable `record_id` (`<slug>-goal-<goal_id:02d>`), the
task slug, the frozen goal identity, success, steps, elapsed time, and termination/truncation flags. A
worker returns exactly its item's episode count and the upstream per-task success rate as
`upstream_task_success_rate` (the current slug only).

A formal `mt50` aggregate validates that all 50 slugs produced 50 unique goal records each, that each
task's record success count matches its upstream `upstream_task_success_rate`, and that no goal is missing
or duplicated. It then reports:

- `metric = "difficulty_macro_success_rate"`;
- per-task success rates in `tasks[]` (`task`, `difficulty`, `episodes`, `success_count`, `success_rate`);
- `difficulty_rates` for `easy`, `medium`, `hard`, `very_hard` (task counts 28 / 11 / 6 / 5);
- `score = (easy + medium + hard + very_hard) / 4`;
- `task_mean_success_rate = success_count / 2500` as a diagnostic only;
- `difficulty_mapping` with the frozen LeRobot mapping version
  `lerobot-difficulty-to-tasks-2026-09-18` and its RFC 8785 SHA-256.

Within each difficulty bucket the task success rates are weighted equally; the four buckets are then
weighted equally, so each bucket carries 25% of the main score. The four buckets are not weighted by task
count. For example, all-easy success with everything else failing scores 25% while the diagnostic
all-task mean is 56%. The upstream cross-task `mean_success_rate` (the Farama all-task average) is not the
platform main score and is never used for ranking, qualification, or the displayed main score.

Quick Test and single-task smoke runs do not produce a formal macro score. A formal score requires all 50
tasks and all four difficulty buckets; missing, duplicate, or failed workers must not be published as a
score and must not shrink the denominator.

## 8. Configuration declaration and precedence

The evaluation image stores the user Policy process contract at the fixed path
`/workspace/roboscope-policy.json` with `schema_version: 2`:

```json
{
  "schema_version": 2,
  "command": ["/opt/user/bin/python", "/workspace/user_policy.py"],
  "cwd": "/workspace",
  "benchmark_config": {"metaworld": {"cameras": [ /* see section 3 */ ]}}
}
```

`benchmark_config` is optional and is returned by image readiness/preflight without importing the Policy
or loading a model. The final evaluation configuration is resolved once, with this priority:

1. the explicit `evaluation_config` supplied for the current run;
2. the image's `benchmark_config.metaworld` declaration;
3. the platform default camera.

The camera list is replaced as a whole; it is never merged by index. Every configured camera then has its
missing fields filled from the platform default and is validated. A request that omits the configuration
(or passes `null`) means "no explicit override"; a provided object must contain a non-empty `cameras` list.
The resolved configuration is written to a task-specific JSON file and passed to the runtime as
`--evaluation-config <path>`; every worker and the Policy `load` session consume that same resolved object,
and the effective configuration is recorded in the result metadata.

The runtime never re-resolves from the image file after the platform has passed a configuration. The format
version `2` is distinct from the benchmark protocol version `metaworld-v1`.

## 9. Validation sequence

Before reporting a MetaWorld score or throughput result:

1. Run startup assertions: 50 slugs, `SawyerXYZEnv.max_path_length == 500`, 39-dimensional observation
   without one-hot, and `Box(4, float32)` action space in `[-1, 1]`.
2. Run the `reach-v3` x 1 training-goal Quick Test through the real policy and image on the target.
3. Run the `reach-v3` x 50 training-goal smoke, confirm 50 unique `goal_id`s, and confirm the same goal
   order and score when repeating with the same seeds.
4. Confirm the delivered image shape, dtype, and processing order match the evaluation configuration, and
   that the default configuration yields 480x480 with a 180-degree rotation.
5. Confirm the upstream per-task success rate matches the aggregated records and that the four difficulty
   rates and the macro `score` follow section 7.
6. Only then run the formal 50 x 50 pass.

## 10. Comparability boundary

RoboScope MetaWorld MT50 is a VLA-oriented evaluation: pixel and language observations, 4-dimensional
delta end-effector actions, and a four-difficulty macro average. It aligns the within-task success rule
with Farama but does not reproduce the official state plus one-hot RL setup. Results must not be compared
with StarVLA, LeRobot, or the paper leaderboard without verifying the protocol (version, task set, goals,
observation, action, horizon, sampling, and aggregation) case by case. LeRobot recommends 10 episodes per
task with its own randomization and observation preprocessing; that is a community protocol, not this
platform's per-training-goal evaluation.

The platform does not write its main score into the paper leaderboard's `average` column. The paper
`average` values are external data with per-paper aggregation that the platform does not control.
