# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Client used by the simulation runtime to call a custom policy process."""

from __future__ import annotations

import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from roboscope_kit import profiling
from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    PolicyCapabilities,
    SessionContext,
)
from roboscope_kit.protocol import (
    ENCODINGS,
    PROTOCOL_VERSION,
    Encoding,
    ProtocolError,
    receive_message,
    send_message,
    validate_version,
)

# Encoding each policy socket agreed to, so per-call clients negotiate once per process.
_NEGOTIATED_ENCODINGS: dict[str, Encoding] = {}


class PolicyRemoteError(RuntimeError):
    def __init__(self, error: dict[str, Any]) -> None:
        self.operation = str(error.get("operation") or "unknown")
        self.error_type = str(error.get("type") or "RemoteError")
        self.remote_message = str(error.get("message") or "")
        self.remote_traceback = str(error.get("traceback") or "")
        self.state = str(error.get("state") or "")
        super().__init__(f"policy {self.operation} failed ({self.error_type}): {self.remote_message}")


class PolicyClient:
    """Client for one policy socket.

    ``encoding`` is the preferred frame encoding; when omitted it is ``ROBOSCOPE_POLICY_ENCODING`` (set by
    the benchmark runtime for its Workers), else ``json``. A preference for ``msgpack`` is used only after
    the server's ``health`` reply lists it; an older server keeps the connection on JSON.
    """

    def __init__(
        self, socket_path: str | Path, timeout_s: float = 30.0, encoding: Encoding | None = None
    ) -> None:
        self.socket_path = str(socket_path)
        self.timeout_s = timeout_s
        preferred = encoding or os.environ.get("ROBOSCOPE_POLICY_ENCODING") or "json"
        if preferred not in ENCODINGS:
            raise ValueError(f"policy encoding must be one of {', '.join(ENCODINGS)}; got {preferred!r}")
        self.preferred_encoding: Encoding = preferred  # type: ignore[assignment]
        self._encoding: Encoding | None = (
            "json" if preferred == "json" else _NEGOTIATED_ENCODINGS.get(self.socket_path)
        )
        self._connection: socket.socket | None = None
        if profiling.ENABLED:
            # Only a --profile Worker pays for the NVTX ranges; otherwise requests run unwrapped.
            self._request = self._profiled_request  # type: ignore[method-assign]

    def connect(self) -> None:
        if self._connection is not None:
            return
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_s)
        connection.connect(self.socket_path)
        self._connection = connection

    def wait_until_ready(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                self.connect()
                return self.health()
            except OSError as exc:
                last_error = exc
                self.close()
                time.sleep(0.05)
        raise TimeoutError(
            f"policy socket {self.socket_path} was not ready within {timeout_s}s"
        ) from last_error

    def health(self) -> dict[str, Any]:
        return self._request("health", {})

    def load(self, context: SessionContext) -> PolicyCapabilities:
        result = self._request("load", {"context": context.to_dict()})
        return PolicyCapabilities.from_dict(result["capabilities"])

    def reset(self, context: EpisodeContext) -> dict[str, Any]:
        if profiling.ENABLED:
            profiling.episode_start()
        return self._request("reset", {"context": context.to_dict()})

    def act(self, observation: Observation) -> ActionChunk:
        result = self._request("act", {"observation": observation.to_dict()})
        return ActionChunk.from_dict(result["action_chunk"])

    def observe(self, observation: Observation) -> None:
        self._request("observe", {"observation": observation.to_dict()})

    def act_batch(self, observations: tuple[Observation, ...]) -> tuple[ActionChunk, ...]:
        result = self._request(
            "act_batch",
            {"observations": [observation.to_dict() for observation in observations]},
        )
        return tuple(ActionChunk.from_dict(item) for item in result["action_chunks"])

    def end_episode(self, episode_id: str) -> dict[str, Any]:
        if not episode_id:
            raise ValueError("episode_id cannot be empty")
        result = self._request("end_episode", {"episode_id": episode_id})
        if profiling.ENABLED:
            profiling.episode_end()
        return result

    def unload(self) -> dict[str, Any]:
        return self._request("unload", {})

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> PolicyClient:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _profiled_request(self, operation: str, payload: dict[str, Any]) -> Any:
        with profiling.request(operation):
            return PolicyClient._request(self, operation, payload)

    @property
    def encoding(self) -> Encoding | None:
        """The frame encoding in use, or ``None`` before a preferred non-JSON encoding is negotiated."""

        return self._encoding

    def _negotiate(self) -> Encoding:
        result = self._exchange("health", {"encodings": [self.preferred_encoding]}, "json")
        accepted = result.get("encodings") if isinstance(result, dict) else None
        encoding: Encoding = (
            self.preferred_encoding if isinstance(accepted, list) and self.preferred_encoding in accepted
            else "json"
        )
        _NEGOTIATED_ENCODINGS[self.socket_path] = encoding
        return encoding

    def _request(self, operation: str, payload: dict[str, Any]) -> Any:
        if self._encoding is None:
            self._encoding = self._negotiate()
        return self._exchange(operation, payload, self._encoding)

    def _exchange(self, operation: str, payload: dict[str, Any], encoding: Encoding) -> Any:
        if self._connection is None:
            self.connect()
        if self._connection is None:
            raise RuntimeError("policy connection is unavailable")
        request_id = str(uuid.uuid4())
        send_message(
            self._connection,
            {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "operation": operation,
                "payload": payload,
            },
            encoding,
        )
        response = receive_message(self._connection)
        if response is None:
            raise ProtocolError("policy server closed the connection without a response")
        validate_version(response)
        if response.get("request_id") != request_id:
            raise ProtocolError("response request_id does not match request")
        if not response.get("ok"):
            raise PolicyRemoteError(response.get("error") or {})
        return response.get("result")
