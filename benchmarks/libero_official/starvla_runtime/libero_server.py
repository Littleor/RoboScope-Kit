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

"""Run the official pool of StarVLA websocket policy-server replicas."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
STARVLA_CHECKPOINT_SERVER = Path(__file__).with_name("starvla_checkpoint_server.py")


@dataclass(frozen=True)
class Replica:
    """One StarVLA policy-server process."""

    index: int
    gpu_id: int
    port: int
    pid: int
    log_path: str


@dataclass
class ReplicaProcess:
    process: subprocess.Popen
    replica: Replica
    log_file: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--starvla-dir", type=Path, default=Path("/root/starVLA"))
    parser.add_argument(
        "--starvla-python",
        type=Path,
        default=Path("/root/miniconda3/envs/starVLA/bin/python"),
    )
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--initial-replicas", type=int)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=16694)
    parser.add_argument("--server-cpu-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=float, default=900)
    return parser.parse_args()


def wait_port(port: int, process: subprocess.Popen, deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited before policy server port {port} became ready")
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"policy server port {port} did not become ready")


def start_replica(
    args: argparse.Namespace,
    index: int,
) -> ReplicaProcess:
    gpu_id = index % args.gpu_count
    port = args.base_port + index
    log_path = args.output_dir / f"policy-server-{index}.log"
    log_file = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{args.starvla_dir}:{REPOSITORY_ROOT}:{env.get('PYTHONPATH', '')}"
    thread_count = str(args.server_cpu_threads)
    env["OMP_NUM_THREADS"] = thread_count
    env["MKL_NUM_THREADS"] = thread_count
    env["OPENBLAS_NUM_THREADS"] = thread_count
    env["NUMEXPR_NUM_THREADS"] = thread_count
    command = [
        str(args.starvla_python),
        str(STARVLA_CHECKPOINT_SERVER),
        "--ckpt_path",
        str(args.checkpoint),
        "--port",
        str(port),
        "--use_bf16",
        "--idle_timeout",
        "-1",
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=args.starvla_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        log_file.close()
        raise
    replica = Replica(
        index=index,
        gpu_id=gpu_id,
        port=port,
        pid=process.pid,
        log_path=str(log_path),
    )
    return ReplicaProcess(process=process, replica=replica, log_file=log_file)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def write_manifest(path: Path, replicas: list[Replica], checkpoint: Path, max_replicas: int) -> None:
    payload = {
        "state": "ready",
        "checkpoint": str(checkpoint),
        "active_replicas": len(replicas),
        "max_replicas": max_replicas,
        "replicas": [asdict(replica) for replica in replicas],
        "written_at_unix_ns": time.time_ns(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def scale_replicas(
    args: argparse.Namespace,
    active: dict[int, ReplicaProcess],
    target: int,
) -> None:
    if not 1 <= target <= args.replicas:
        raise ValueError(f"replica target must be between 1 and {args.replicas}")
    for index in sorted((index for index in active if index >= target), reverse=True):
        replica_process = active.pop(index)
        stop_process(replica_process.process)
        replica_process.log_file.close()

    started: list[ReplicaProcess] = []
    for index in range(target):
        if index in active:
            continue
        replica_process = start_replica(args, index)
        active[index] = replica_process
        started.append(replica_process)
    deadline = time.monotonic() + args.ready_timeout
    for replica_process in started:
        wait_port(replica_process.replica.port, replica_process.process, deadline)
        if replica_process.process.poll() is not None:
            raise RuntimeError(
                f"policy server {replica_process.replica.index} exited with "
                f"{replica_process.process.returncode}"
            )


def read_scale_target(path: Path, current: int) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return current
    return int(payload["target_replicas"])


def main() -> None:
    args = parse_args()
    if args.replicas < 1:
        raise ValueError("replicas must be positive")
    initial_replicas = args.replicas if args.initial_replicas is None else args.initial_replicas
    if not 1 <= initial_replicas <= args.replicas:
        raise ValueError("initial-replicas must be between 1 and replicas")
    if args.gpu_count < 1:
        raise ValueError("gpu-count must be positive")
    if args.server_cpu_threads < 1:
        raise ValueError("server-cpu-threads must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    active: dict[int, ReplicaProcess] = {}
    manifest_path = args.output_dir / "server-manifest.json"
    control_path = args.output_dir / "server-control.json"
    try:
        scale_replicas(args, active, initial_replicas)
        write_manifest(
            manifest_path,
            [item.replica for _, item in sorted(active.items())],
            args.checkpoint,
            args.replicas,
        )

        while not stop_requested:
            target = read_scale_target(control_path, len(active))
            if target != len(active):
                scale_replicas(args, active, target)
                write_manifest(
                    manifest_path,
                    [item.replica for _, item in sorted(active.items())],
                    args.checkpoint,
                    args.replicas,
                )
            for replica_process in active.values():
                if replica_process.process.poll() is not None:
                    raise RuntimeError(
                        f"policy server {replica_process.replica.index} exited with "
                        f"{replica_process.process.returncode}"
                    )
            time.sleep(0.2)
    finally:
        for replica_process in active.values():
            stop_process(replica_process.process)
            replica_process.log_file.close()


if __name__ == "__main__":
    main()
