# Copyright 2026 RoboScope Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one official StarVLA LIBERO client for a measurement window."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.libero_official.starvla_runtime.libero_policy_stepper import (
    PolicyStepper,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--warmup-action-steps", type=int, default=8)
    parser.add_argument("--barrier-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def binarize_gripper(open_value: np.ndarray | float) -> np.ndarray:
    value = float(np.asarray(open_value, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator


def build_example(observation: dict) -> dict:
    primary = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"image": [primary, wrist]}


def process_cpu_times() -> tuple[float, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


def native_thread_count() -> int | None:
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


def response_action(response: dict) -> list[float]:
    raw_action = response["raw_action"]
    world = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
    rotation = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
    open_gripper = np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(-1)
    if world.size != 3 or rotation.size != 3 or open_gripper.size != 1:
        raise ValueError(
            "invalid action sizes: "
            f"world={world.shape}, rotation={rotation.shape}, gripper={open_gripper.shape}"
        )
    return np.concatenate((world, rotation, binarize_gripper(open_gripper))).tolist()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_start(path: Path) -> dict:
    while not path.exists():
        time.sleep(0.05)
    return json.loads(path.read_text(encoding="utf-8"))


def episode_terminated(env: object, task_done: bool) -> bool:
    return task_done or bool(env.env.done)


def reset_episode(
    env: object,
    client: object,
    task_description: str,
    initial_states: np.ndarray,
    episode_index: int,
    settle_steps: int,
) -> tuple[dict, int]:
    for _ in range(len(initial_states)):
        client.reset(task_description=task_description)
        env.reset()
        observation = env.set_init_state(initial_states[episode_index])
        terminated = False
        for _ in range(settle_steps):
            observation, _reward, task_done, _info = env.step(LIBERO_DUMMY_ACTION)
            terminated = episode_terminated(env, task_done)
            if terminated:
                episode_index = (episode_index + 1) % len(initial_states)
                break
        if not terminated:
            return observation, episode_index
    raise RuntimeError("all LIBERO initial states terminated during settling")


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed + args.worker_id)
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id = args.task_id % task_suite.n_tasks
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed + args.worker_id)
    client = ModelClient(host=args.host, port=args.port, unnorm_key="franka")
    stepper = PolicyStepper(client, build_example)
    try:
        episode_index = args.episode_index % len(initial_states)
        observation, episode_index = reset_episode(
            env,
            stepper,
            task.language,
            initial_states,
            episode_index,
            args.settle_steps,
        )
        for step in range(args.warmup_action_steps):
            policy_step = stepper.step(observation, task.language, step)
            observation, _reward, done, _info = env.step(response_action(policy_step.response))
            if episode_terminated(env, done):
                break

        observation, episode_index = reset_episode(
            env,
            stepper,
            task.language,
            initial_states,
            episode_index,
            args.settle_steps,
        )
        ready_path = args.barrier_dir / f"ready-{args.worker_id}.json"
        atomic_json(
            ready_path,
            {
                "worker_id": args.worker_id,
                "task_id": task_id,
                "port": args.port,
                "ready_at_unix_ns": time.time_ns(),
            },
        )
        start = wait_for_start(args.barrier_dir / "start.json")
        target_start_ns = int(start["start_at_unix_ns"])
        duration_s = float(start["duration_s"])
        while time.time_ns() < target_start_ns:
            time.sleep(0.001)

        started_at_ns = time.time_ns()
        started_at = time.perf_counter()
        user_cpu_start_s, system_cpu_start_s = process_cpu_times()
        thread_count_start = native_thread_count()
        deadline = started_at + duration_s
        steps = 0
        inference_requests = 0
        inference_wait_s = 0.0
        observation_build_s = 0.0
        client_step_s = 0.0
        action_decode_s = 0.0
        env_step_s = 0.0
        environment_step_s = 0.0
        reset_s = 0.0
        inference_latencies_ms: list[float] = []
        episodes = 0
        successes = 0
        episode_step = 0
        while time.perf_counter() < deadline:
            policy_step = stepper.step(observation, task.language, episode_step)
            observation_build_s += policy_step.observation_build_s
            inference_wait_s += policy_step.blocking_client_step_s
            client_step_s += policy_step.blocking_client_step_s
            if policy_step.request_latency_ms is not None:
                inference_requests += 1
                inference_latencies_ms.append(policy_step.request_latency_ms)

            environment_started = time.perf_counter()
            action_started = time.perf_counter()
            action = response_action(policy_step.response)
            action_decode_s += time.perf_counter() - action_started
            env_started = time.perf_counter()
            observation, _reward, done, _info = env.step(action)
            env_step_s += time.perf_counter() - env_started
            environment_step_s += time.perf_counter() - environment_started
            steps += 1
            episode_step += 1
            if episode_terminated(env, done):
                episodes += 1
                successes += int(done)
                episode_index = (episode_index + 1) % len(initial_states)
                reset_started = time.perf_counter()
                observation, episode_index = reset_episode(
                    env,
                    stepper,
                    task.language,
                    initial_states,
                    episode_index,
                    args.settle_steps,
                )
                reset_s += time.perf_counter() - reset_started
                episode_step = 0

        ended_at_ns = time.time_ns()
        ended_at = time.perf_counter()
        user_cpu_end_s, system_cpu_end_s = process_cpu_times()
        thread_count_end = native_thread_count()
        elapsed_s = max(ended_at - started_at, 1e-9)
        user_cpu_s = max(user_cpu_end_s - user_cpu_start_s, 0.0)
        system_cpu_s = max(system_cpu_end_s - system_cpu_start_s, 0.0)
        process_cpu_s = user_cpu_s + system_cpu_s
        atomic_json(
            args.result,
            {
                "worker_id": args.worker_id,
                "task_id": task_id,
                "port": args.port,
                "steps": steps,
                "inference_requests": inference_requests,
                "inference_wait_s": inference_wait_s,
                "client_step_s": client_step_s,
                "observation_build_s": observation_build_s,
                "action_decode_s": action_decode_s,
                "env_step_s": env_step_s,
                "environment_step_s": environment_step_s,
                "reset_s": reset_s,
                "inference_latencies_ms": inference_latencies_ms,
                "episodes": episodes,
                "successes": successes,
                "started_at_unix_ns": started_at_ns,
                "ended_at_unix_ns": ended_at_ns,
                "elapsed_s": elapsed_s,
                "process_user_cpu_s": user_cpu_s,
                "process_system_cpu_s": system_cpu_s,
                "process_cpu_s": process_cpu_s,
                "process_cpu_cores": process_cpu_s / elapsed_s,
                "native_thread_count_start": thread_count_start,
                "native_thread_count_end": thread_count_end,
            },
        )
    finally:
        client.client.close()
        env.close()


if __name__ == "__main__":
    main()
