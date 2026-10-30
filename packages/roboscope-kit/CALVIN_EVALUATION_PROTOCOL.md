# CALVIN Evaluation Protocol

This document defines the SDK-native RoboScope CALVIN evaluation contract. The platform owns benchmark
enumeration, simulator processes, concurrency, recovery, and result aggregation. User inference code owns
only the public `Policy` lifecycle and model-specific preprocessing and postprocessing.

SDK 0.7.0b58 and newer require Python 3.11 or newer in every process that
imports the SDK, including Policy and simulator workers. Python 3.8–3.10
runtimes are unsupported; existing images require separate upgrade and
acceptance before adopting this release. See the [SDK Python policy](README.md#python-compatibility).

## 1. Benchmark scope

The supported suites are the official CALVIN challenge splits `task_D_D`, `task_ABC_D`, and
`task_ABCD_D`. The selected suite declares the model's training split and all three evaluate on the
official environment D challenge. An official evaluation contains 1,000 ordered
long-horizon sequences. Each sequence contains one initial condition followed by five language subtasks.
The score is the number of consecutive subtasks completed before the first failure.

The platform reads the ordered sequence list from `eval_sequences.json`, requires exactly 1,000 entries and
the official SHA-256 `c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317`, records
that digest in the runtime manifest, and passes it to every evaluator worker. A worker exits with an error
if the file changes before evaluation starts. Probe runs use a fixed prefix of the same manifest; an
official score uses all 1,000 entries.

## 2. User policy lifecycle

The user entrypoint implements `Policy` from `roboscope_kit` and calls `run(policy)`. RoboScope
starts one process per selected replica and injects a Unix socket through `ROBOSCOPE_INFERENCE_SOCKET`.
The platform then calls:

1. `Policy.load(SessionContext)` once per replica;
2. `Policy.reset(EpisodeContext)` before each language subtask;
3. `Policy.act(Observation)` whenever the evaluator needs a new action chunk;
4. after a successful reset, `Policy.end_episode(episode_id)` when each subtask finishes or aborts;
5. `Policy.unload()` when evaluation ends.

The Policy process owns mutable episode state and should key it by `episode_id`. The SDK does not store or
synchronize user state.

`PolicyCapabilities` must declare CALVIN, the user-selected suite, every required feature, and supported
action-chunk sizes. Evaluator concurrency and Policy replica count are selected by the platform runtime.

Real model loading can exceed 30 seconds under cold-cache or shared-storage conditions. Platform
controllers may therefore configure a longer `PolicyProcessAdapter.load_timeout_s`; the SDK default remains
unchanged for existing integrations. The reference StarVLA controller uses a measured 120-second
per-replica load timeout, independently of the overall replica and worker readiness timeout.

## 3. Canonical observation

Every `Observation` contains the following benchmark-native features:

| Feature | Type | Meaning |
| --- | --- | --- |
| `images.static` | contiguous `uint8` array | CALVIN static RGB camera image |
| `images.gripper` | contiguous `uint8` array | CALVIN gripper RGB camera image |
| `state.proprioception` | contiguous `float32` array | Native CALVIN robot observation vector |
| `language.instruction` | `str` | Official validation annotation for the current subtask |

The platform does not resize, normalize, transpose, or otherwise adapt images for a particular model.
Those transformations belong inside the user policy. `episode_id` identifies the sequence and subtask;
`step` starts at zero for each subtask and increases once per simulator action.

## 4. Action contract

`Policy.act()` returns an `ActionChunk` containing either one 7-dimensional action or a two-dimensional
array of 7-dimensional actions. The seven values are the native CALVIN relative end-effector action:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

The gripper command supplied to the simulator must use CALVIN's native `-1` or `+1` representation. A
model that predicts a continuous gripper value must convert it explicitly in the policy. The evaluator
consumes the returned chunk in order and requests another chunk only when the current one is empty. Cached
actions are discarded at every subtask reset.

## 5. Official rollout behavior

The persistent evaluator worker imports `calvin_agent.evaluation.evaluate_policy.evaluate_sequence` and
uses the official task oracle, validation annotations, environment reset, subtask horizon, action stepping,
and success detection. RoboScope does not reimplement those rules.

Before a sequence starts, the worker selects its initial condition and five-task list by the exact manifest
index. The official evaluator resets the environment to that initial condition. It then evaluates subtasks
in order and stops the sequence at the first failure. The worker wraps `env.step` only to count simulator
steps for throughput reporting; the action and return value are unchanged.

Unexpected simulator, protocol, or policy errors terminate the worker and surface through the generic
runtime. They are not converted into failed tasks.

## 6. Work manifest and recovery

Each `WorkItem` contains a start sequence index and an exact sequence count. It declares the same number of
expected terminal records. Every completed sequence produces exactly one record:

```text
sequence-0000
sequence-0001
...
sequence-0999
```

The record metadata contains the sequence index, ordered task list, and chained success length. Final
aggregation rejects missing indices, duplicate indices, unexpected record IDs, or coverage outside the
requested prefix. Generic queue recovery may rerun an uncommitted work item after a worker failure, but it
cannot silently skip or double-count a sequence.

## 7. Official aggregation

For chained success lengths `L_i` in `[0, 5]`, the result contains:

- `avg_seq_len`: the mean of `L_i`;
- `task_1` through `task_5`: the fraction of sequences with `L_i >= 1` through `L_i >= 5`;
- `success_at_5`: the fraction completing all five subtasks;
- `sequence_success_lengths`: the manifest-ordered list of all `L_i` values.

The generic runtime additionally reports total simulator steps, evaluation wall time, steps per second,
episodes per second, resource samples, and worker-recovery events. These operational metrics do not replace
the official CALVIN aggregation.

## 8. Validation sequence

Before reporting a score or throughput result:

1. Verify that the sequence manifest contains the official ordered 1,000 entries and record its SHA-256.
2. Run one sequence with K=1 and g=1 through the real policy and simulator.
3. Run a small fixed parallel prefix and compare its per-sequence chained lengths with the established
   execution path.
4. Confirm exact record coverage, an empty pending/running queue, no worker failures, and no OOM recovery.
5. Tune K and g only with the same fixed prefix for every candidate.
6. Run all 1,000 sequences once with the selected fixed K/g configuration.

The image-provided Policy keeps image preprocessing and gripper conversion model-side; the CALVIN
adapter itself contains no model-specific imports.
