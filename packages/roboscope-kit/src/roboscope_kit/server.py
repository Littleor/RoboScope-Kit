# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Unix socket server that exposes a user Policy to the simulation runtime."""

from __future__ import annotations

import os
import socket
import threading
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    validate_action_chunk,
    validate_observation,
    validate_policy_capabilities,
)
from roboscope_kit.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    receive_message,
    send_message,
    validate_version,
)

DEFAULT_SOCKET_PATH = "/tmp/roboscope-inference.sock"


class PolicyState(str, Enum):  # noqa: UP042 -- benchmark environments include Python 3.10
    CREATED = "created"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADING = "unloading"
    CLOSED = "closed"


class PolicyServer:
    def __init__(self, policy: Policy, socket_path: str | Path) -> None:
        self.policy = policy
        self.socket_path = Path(socket_path)
        self.state = PolicyState.CREATED
        self.context: SessionContext | None = None
        self.capabilities: PolicyCapabilities | None = None
        self._request_lock = threading.Lock()

    def serve(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(self.socket_path))
                self.socket_path.chmod(0o600)
                listener.listen(64)
                listener.settimeout(0.2)
                while self.state is not PolicyState.CLOSED:
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    threading.Thread(
                        target=self._serve_client,
                        args=(connection,),
                        daemon=True,
                    ).start()
        finally:
            self._close_policy()
            self.socket_path.unlink(missing_ok=True)

    def _serve_client(self, connection: socket.socket) -> None:
        with connection:
            self._serve_connection(connection)

    def _serve_connection(self, connection: socket.socket) -> None:
        while self.state is not PolicyState.CLOSED:
            request: dict[str, Any] | None = None
            try:
                request = receive_message(connection)
                if request is None:
                    return
                with self._request_lock:
                    response = self._handle(request)
            except Exception as exc:
                request_id = str(request.get("request_id") or "") if request is not None else ""
                response = self._error_response(exc, request_id=request_id)
            send_message(connection, response)

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or uuid.uuid4())
        operation = str(request.get("operation") or "")
        try:
            validate_version(request)
            result = self._dispatch(operation, request.get("payload") or {})
            return self._success_response(request_id, result)
        except Exception as exc:
            if operation in {"load", "reset", "act", "act_batch", "end_episode"} and not isinstance(
                exc, ProtocolError
            ):
                self.state = PolicyState.FAILED
            return self._error_response(exc, request_id=request_id, operation=operation)

    def _dispatch(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"state": self.state.value}
        if operation == "load":
            self._require_state(operation, PolicyState.CREATED)
            self.state = PolicyState.LOADING
            context = SessionContext.from_dict(payload["context"])
            capabilities = self.policy.load(context)
            if not isinstance(capabilities, PolicyCapabilities):
                raise TypeError("Policy.load() must return PolicyCapabilities")
            validate_policy_capabilities(context.benchmark, capabilities)
            self.context = context
            self.capabilities = capabilities
            self.state = PolicyState.READY
            return {"state": self.state.value, "capabilities": capabilities.to_dict()}
        if operation == "reset":
            self._require_state(operation, PolicyState.READY)
            self.policy.reset(EpisodeContext.from_dict(payload["context"]))
            return {"state": self.state.value}
        if operation == "act":
            self._require_state(operation, PolicyState.READY)
            context, capabilities = self._loaded_contract()
            observation = Observation.from_dict(payload["observation"])
            validate_observation(observation, context.benchmark, capabilities)
            action = self.policy.act(observation)
            if not isinstance(action, ActionChunk):
                raise TypeError("Policy.act() must return ActionChunk")
            validate_action_chunk(action, context.benchmark)
            self._validate_action_chunk_size(action, capabilities)
            return {"action_chunk": action.to_dict()}
        if operation == "act_batch":
            self._require_state(operation, PolicyState.READY)
            context, capabilities = self._loaded_contract()
            observations = tuple(Observation.from_dict(item) for item in payload["observations"])
            if not observations:
                raise ValueError("observation batch cannot be empty")
            if len(observations) > capabilities.max_batch_size:
                raise ValueError(
                    f"observation batch size {len(observations)} exceeds policy maximum "
                    f"{capabilities.max_batch_size}"
                )
            for observation in observations:
                validate_observation(observation, context.benchmark, capabilities)
            actions = self.policy.act_batch(observations)
            if len(actions) != len(observations):
                raise ValueError("Policy.act_batch() must return one ActionChunk per observation")
            for action in actions:
                if not isinstance(action, ActionChunk):
                    raise TypeError("Policy.act_batch() must return ActionChunk values")
                validate_action_chunk(action, context.benchmark)
                self._validate_action_chunk_size(action, capabilities)
            return {"action_chunks": [action.to_dict() for action in actions]}
        if operation == "end_episode":
            self._require_state(operation, PolicyState.READY)
            episode_id = str(payload.get("episode_id") or "")
            if not episode_id:
                raise ValueError("episode_id cannot be empty")
            self.policy.end_episode(episode_id)
            return {"state": self.state.value}
        if operation == "unload":
            self._require_state(operation, PolicyState.CREATED, PolicyState.READY, PolicyState.FAILED)
            self._close_policy()
            return {"state": self.state.value}
        raise ProtocolError(f"unknown operation {operation!r}")

    def _close_policy(self) -> None:
        if self.state in {PolicyState.UNLOADING, PolicyState.CLOSED}:
            return
        should_unload = self.state is not PolicyState.CREATED
        self.state = PolicyState.UNLOADING
        try:
            if should_unload:
                self.policy.unload()
        finally:
            self.state = PolicyState.CLOSED

    def _require_state(self, operation: str, *allowed: PolicyState) -> None:
        if self.state not in allowed:
            expected = ", ".join(state.value for state in allowed)
            raise ProtocolError(
                f"operation {operation!r} is invalid in state {self.state.value!r}; expected {expected}"
            )

    def _loaded_contract(self) -> tuple[SessionContext, PolicyCapabilities]:
        if self.context is None or self.capabilities is None:
            raise RuntimeError("policy contract is unavailable before load")
        return self.context, self.capabilities

    @staticmethod
    def _validate_action_chunk_size(
        action: ActionChunk,
        capabilities: PolicyCapabilities,
    ) -> None:
        chunk_size = 1 if action.actions.ndim == 1 else action.actions.shape[0]
        if chunk_size not in capabilities.supported_action_chunk_sizes:
            raise ValueError(
                f"policy returned action chunk size {chunk_size}; supported sizes are "
                f"{capabilities.supported_action_chunk_sizes}"
            )

    def _success_response(self, request_id: str, result: Any) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": True,
            "result": result,
        }

    def _error_response(
        self,
        exc: Exception,
        *,
        request_id: str,
        operation: str = "protocol",
    ) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": False,
            "error": {
                "operation": operation,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "state": self.state.value,
            },
        }


def run(policy: Policy) -> None:
    """Run a policy at the socket path injected by the RoboScope runtime."""

    socket_path = os.environ.get("ROBOSCOPE_INFERENCE_SOCKET", DEFAULT_SOCKET_PATH)
    PolicyServer(policy, socket_path).serve()
