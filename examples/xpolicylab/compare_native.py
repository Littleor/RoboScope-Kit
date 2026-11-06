# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Compare a RoboScope Kit RoboDojo run with the pinned upstream native entry point.

    python compare_native.py --runtime /sim/robodojo --policy-env /sim/robodojo/miniforge/envs/simate-beta-eval \\
        --checkpoint /models/simate-beta-robodojo-eval --kit-run /runs/simate-arx_x5 --output /runs/simate-native

For every task in the Kit run it evaluates the same layouts upstream, unmodified: XPolicyLab's
``setup_eval_policy_server.sh`` serves ``--policy`` from ``--policy-env``, and RoboDojo's own
``src/eval_client/main.py --num_envs 1`` runs ``EVAL_NUM`` episodes with seed 1, which are the
task's first layouts, as the Kit's ``--episodes`` prefix is. The upstream code is a copy of the
build's RoboDojo checkout under ``<output>/RoboDojo`` with the Kit patches reverted
(``git checkout -- .`` in RoboDojo and XPolicyLab), so it runs on the build's simulator
environment, assets, and installed submodules.

Episodes are paired by (task, layout) and compared on success and executed steps. Upstream does not
report steps, so they are read from the frame count of the episode's head-camera video. Upstream's
client can exit with status 0 after an unhandled exception, so a task counts as complete only when its
``_result.json`` holds every requested episode.

Writes ``<output>/tasks.jsonl`` (per-task status and wall time), ``<output>/comparison.json``, and
one server and client log per task under ``<output>/logs/``. A rerun skips tasks that already
completed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

POLICY_PORT = 59100
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FRAMES = re.compile(r"episode_(\d+)_cam_head_\w+\.mp4`, containing (\d+) frames")


def kit_records(run: Path) -> dict[tuple[str, int], dict]:
    records = {}
    for path in sorted((run / "queue" / "completed").glob("*.json")):
        for record in json.loads(path.read_text())["records"]:
            key = (record["metadata"]["task"], int(record["metadata"]["layout_id"]))
            if key in records:
                raise SystemExit(f"duplicate Kit record {key} in {path}")
            records[key] = {
                "success": bool(record["success"]),
                "steps": int(record["steps"]),
                "score": record["metadata"].get("score"),
            }
    if not records:
        raise SystemExit(f"{run} has no completed Kit records")
    return records


def prepare_checkout(runtime: Path, output: Path) -> Path:
    checkout = output / "RoboDojo"
    if not checkout.exists():
        print(f"copying {runtime / 'RoboDojo'} to {checkout}", flush=True)
        shutil.copytree(runtime / "RoboDojo", checkout, symlinks=True, ignore=shutil.ignore_patterns("eval_result"))
        for repo in (checkout, checkout / "XPolicyLab"):
            subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", repo, "checkout", "--", "."], check=True)
    state = []
    for repo in (checkout, checkout / "XPolicyLab"):
        git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
        state.append(subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout)
        state.append(subprocess.run([*git, "status", "--short", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout)
    (output / "source-state.txt").write_text("".join(state))
    return checkout


def wait_for_port(port: int, server: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def run_task(args, checkout: Path, task: str, episodes: int, port: int) -> dict:
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_name = f"native-{task}"
    result_dir = checkout / "eval_result/RoboDojo" / task / args.policy / "arx_x5" / f"1_{run_name}"
    shutil.rmtree(result_dir, ignore_errors=True)
    env = {
        **os.environ,
        "ACCEPT_EULA": "Y",
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONPATH": os.pathsep.join((str(checkout), str(checkout / "XPolicyLab"))),
        "PYTHONNOUSERSITE": "1",
        "EVAL_NUM": str(episodes),
        "ROBODOJO_RUN_ID": run_name,
    }
    for name in ("ROBODOJO_LAYOUT_IDS", "ROBODOJO_ASYNC_VIDEO", "ROBODOJO_UNPACED_RENDER",
                 "ROBODOJO_CONTROL_STATE_REUSE", "ROBODOJO_PHYSICS_THREADS"):
        env.pop(name, None)
    started = time.time()
    with open(logs / f"{task}-server.log", "w") as server_log:
        server = subprocess.Popen(
            ["bash", f"XPolicyLab/policy/{args.policy}/setup_eval_policy_server.sh", "RoboDojo", task,
             str(args.checkpoint), "arx_x5", args.action_type, "1", "0", str(args.policy_env), str(port), "127.0.0.1"],
            cwd=checkout, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        ready = wait_for_port(port, server, args.server_timeout)
        server_ready = time.time()
        returncode = None
        if ready:
            with open(logs / f"{task}-client.log", "w") as client_log:
                returncode = subprocess.run(
                    [str(args.runtime / "env/bin/python"), "-u", "src/eval_client/main.py", "--task_name", task,
                     "--env_cfg_type", "arx_x5", "--num_envs", "1", "--enable_cameras", "--kit_args",
                     "--allow-root --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
                     "--device_id", "0", "--policy_name", args.policy, "--port", str(port), "--protocol", "ws",
                     "--policy_server_url", f"ws://127.0.0.1:{port}", "--additional_info", run_name,
                     "--seed", "1", "--host", "127.0.0.1", "--headless"],
                    cwd=checkout, env=env, stdout=client_log, stderr=subprocess.STDOUT,
                    timeout=args.task_timeout, check=False,
                ).returncode
    finally:
        try:
            os.killpg(server.pid, 15)
        except ProcessLookupError:
            pass
        server.wait()
    results = sorted(result_dir.glob("*/_result.json"))
    status = {"task": task, "server_ready_s": server_ready - started, "wall_s": time.time() - started,
              "client_returncode": returncode, "complete": False}
    if not ready:
        status["error"] = "policy server did not become ready"
    elif len(results) != 1:
        status["error"] = f"expected one _result.json, found {len(results)}"
    else:
        result = json.loads(results[0].read_text())
        if result.get("eval_time") != episodes:
            status["error"] = f"_result.json holds {result.get('eval_time')} of {episodes} episodes"
        else:
            log = ANSI.sub("", (logs / f"{task}-client.log").read_text(errors="replace"))
            frames = {int(index): int(count) for index, count in FRAMES.findall(log)}
            status.update(complete=True, episodes=[
                {"layout_id": int(item["layout_id"]), "success": bool(item["success"]), "score": item.get("score"),
                 "steps": frames[int(index)] - 1 if int(index) in frames else None}
                for index, item in sorted(result["details"].items(), key=lambda kv: int(kv[0]))
            ])
    return status


def mcnemar(only_kit: int, only_native: int) -> float:
    n = only_kit + only_native
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_kit, only_native) + 1))
    return min(1.0, 2 * tail / 2**n)


def compare(kit: dict, native: dict) -> dict:
    keys = sorted(set(kit) | set(native))
    paired = [key for key in keys if key in kit and key in native]
    only_kit = sum(1 for key in paired if kit[key]["success"] and not native[key]["success"])
    only_native = sum(1 for key in paired if native[key]["success"] and not kit[key]["success"])
    with_steps = [key for key in paired if native[key]["steps"] is not None]
    tasks = {}
    for task, layout in paired:
        entry = tasks.setdefault(task, {"episodes": 0, "kit_successes": 0, "native_successes": 0})
        entry["episodes"] += 1
        entry["kit_successes"] += kit[(task, layout)]["success"]
        entry["native_successes"] += native[(task, layout)]["success"]
    return {
        "paired_episodes": len(paired),
        "missing_native": [list(key) for key in keys if key not in native],
        "missing_kit": [list(key) for key in keys if key not in kit],
        "kit_successes": sum(kit[key]["success"] for key in paired),
        "native_successes": sum(native[key]["success"] for key in paired),
        "only_kit_success": only_kit,
        "only_native_success": only_native,
        "mcnemar_exact_p": mcnemar(only_kit, only_native),
        "steps_compared": len(with_steps),
        "steps_equal_within_1": sum(1 for key in with_steps if abs(kit[key]["steps"] - native[key]["steps"]) <= 1),
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runtime", type=Path, required=True, help="build-runtime.sh install directory")
    parser.add_argument("--policy-env", type=Path, required=True, help="the policy's Conda environment")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-run", type=Path, required=True, help="the Kit run's --output-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="Simate_beta")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--task", action="append", help="only these tasks (repeatable); default every Kit task")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--server-timeout", type=float, default=600.0)
    parser.add_argument("--task-timeout", type=float, default=4 * 3600.0)
    args = parser.parse_args()
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"YES", "Y", "1", "TRUE"}:
        raise SystemExit("Read the Isaac Sim license and set OMNI_KIT_ACCEPT_EULA=YES to accept it.")
    args.output.mkdir(parents=True, exist_ok=True)
    args.runtime, args.policy_env, args.checkpoint = (
        args.runtime.resolve(), args.policy_env.resolve(), args.checkpoint.resolve())

    kit = kit_records(args.kit_run)
    layouts: dict[str, list[int]] = {}
    for task, layout in kit:
        layouts.setdefault(task, []).append(layout)
    for task, ids in layouts.items():
        if sorted(ids) != list(range(len(ids))):
            raise SystemExit(f"Kit task {task} has layouts {sorted(ids)}; upstream EVAL_NUM can only run a prefix")
    tasks = args.task or list(dict.fromkeys(task for task, _ in kit))
    unknown = sorted(set(tasks) - set(layouts))
    if unknown:
        raise SystemExit(f"tasks not in the Kit run: {', '.join(unknown)}")
    checkout = prepare_checkout(args.runtime, args.output)

    status_path = args.output / "tasks.jsonl"
    done = {}
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            status = json.loads(line)
            if status.get("complete"):
                done[status["task"]] = status
    for offset, task in enumerate(tasks):
        if task in done:
            continue
        print(f"[{offset + 1}/{len(tasks)}] {task}: {len(layouts[task])} episodes", flush=True)
        status = run_task(args, checkout, task, len(layouts[task]), POLICY_PORT + offset)
        with status_path.open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        if status["complete"]:
            done[task] = status
        else:
            print(f"  incomplete: {status['error']}", file=sys.stderr, flush=True)

    native = {(task, episode["layout_id"]): episode for task, status in done.items() for episode in status["episodes"]}
    kit = {key: value for key, value in kit.items() if key[0] in tasks}
    comparison = compare(kit, native)
    comparison["native_wall_s"] = sum(status["wall_s"] for status in done.values())
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=1))
    print(json.dumps({key: value for key, value in comparison.items() if key != "tasks"}, indent=1))
    for task, entry in comparison["tasks"].items():
        if entry["kit_successes"] != entry["native_successes"]:
            print(f"  {task}: Kit {entry['kit_successes']}/{entry['episodes']}, "
                  f"native {entry['native_successes']}/{entry['episodes']}")
    return 0 if not comparison["missing_native"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
