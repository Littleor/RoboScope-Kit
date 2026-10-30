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

"""Action-chunk stepping for the official StarVLA LIBERO client."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyStepResult:
    response: dict
    observation_build_s: float
    blocking_client_step_s: float
    request_latency_ms: float | None


class PolicyStepper:
    """Skip image construction while the client is consuming cached actions."""

    def __init__(
        self,
        client: object,
        build_example: Callable[[dict], dict],
    ) -> None:
        self.client = client
        self._build_example = build_example

    @property
    def action_chunk_size(self) -> int:
        return int(self.client.action_chunk_size)

    def reset(self, task_description: str) -> None:
        self.client.reset(task_description=task_description)

    def step(self, observation: dict, task_description: str, step: int) -> PolicyStepResult:
        needs_request = step % self.action_chunk_size == 0 or self.client.raw_actions is None
        observation_build_s = 0.0
        example = {"lang": task_description}
        if needs_request:
            started = time.perf_counter()
            example.update(self._build_example(observation))
            observation_build_s = time.perf_counter() - started

        started = time.perf_counter()
        response = self.client.step(example=example, step=step)
        blocking_s = time.perf_counter() - started
        request_latency_ms = blocking_s * 1000 if needs_request else None
        return PolicyStepResult(response, observation_build_s, blocking_s, request_latency_ms)
