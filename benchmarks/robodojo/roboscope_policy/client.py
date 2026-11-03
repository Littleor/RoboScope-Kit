# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Connect RoboDojo's evaluation lifecycle to a RoboScope Policy replica."""

from __future__ import annotations

import os
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from roboscope_kit import EpisodeContext, PolicyClient


class WsModelClient:
    """RoboDojo client interface backed by the kit's local Policy protocol."""

    def __init__(self, **_kwargs):
        self.socket_path = os.environ["ROBOSCOPE_INFERENCE_SOCKET"]
        self.episode_id = ""
        self.queue_size = int(os.environ.get("ROBOSCOPE_ROBODOJO_OBSERVATION_QUEUE_SIZE", "0"))
        if not 0 <= self.queue_size <= 10:
            raise ValueError("RoboDojo observation queue size must be between 0 and 10")
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=1) if self.queue_size else None

    def call(self, func_name=None, obs=None, **_kwargs):
        if func_name != "reset" or obs is not None:
            raise ValueError(f"unsupported RoboDojo model operation: {func_name!r}")
        self.end_episode()

    def begin_episode(self, task, layout_id):
        self.end_episode()
        episode_id = f"robodojo-{uuid.uuid4()}"
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            client.reset(EpisodeContext(
                episode_id=episode_id,
                task_id=task,
                metadata={"layout_id": layout_id},
            ))
        self.episode_id = episode_id
        return episode_id

    def act(self, observation):
        self._flush()
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            return client.act(observation).actions

    def _observe_after(self, previous, observation):
        if previous is not None:
            previous.result()
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            client.observe(observation)

    def observe(self, observation):
        if self.executor is None:
            self._observe_after(None, observation)
            return
        while len(self.pending) >= self.queue_size:
            self.pending.popleft().result()
        previous = self.pending[-1] if self.pending else None
        self.pending.append(self.executor.submit(self._observe_after, previous, observation))

    def _flush(self):
        while self.pending:
            self.pending.popleft().result()

    def end_episode(self):
        if self.episode_id:
            episode_id, self.episode_id = self.episode_id, ""
            try:
                self._flush()
            except BaseException as primary:
                try:
                    with PolicyClient(self.socket_path, timeout_s=900) as client:
                        client.end_episode(episode_id)
                except BaseException as cleanup:
                    raise primary from cleanup
                raise
            with PolicyClient(self.socket_path, timeout_s=900) as client:
                client.end_episode(episode_id)

    def close(self):
        try:
            self.end_episode()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
