# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Framing and safe JSON serialization for the local policy socket."""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct(">I")
_TYPE_KEY = "__roboscope_type__"


class ProtocolError(RuntimeError):
    """Raised when a peer sends an invalid or unsupported message."""


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(_encode(message), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message is {len(payload)} bytes; maximum is {MAX_MESSAGE_BYTES}")
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def receive_message(connection: socket.socket) -> dict[str, Any] | None:
    header = _receive_exact(connection, _HEADER.size)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message declares {size} bytes; maximum is {MAX_MESSAGE_BYTES}")
    payload = _receive_exact(connection, size)
    if payload is None:
        raise ProtocolError("connection closed before message body")
    decoded = _decode(json.loads(payload.decode("utf-8")))
    if not isinstance(decoded, dict):
        raise ProtocolError("message root must be an object")
    return decoded


def validate_version(message: dict[str, Any]) -> None:
    version = message.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version!r}; expected {PROTOCOL_VERSION}")


def _receive_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None if remaining == size else b"".join(chunks)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            _TYPE_KEY: "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {_TYPE_KEY: "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("protocol object keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get(_TYPE_KEY)
    if value_type == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if value_type == "ndarray":
        data = base64.b64decode(value["data"], validate=True)
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ProtocolError("object arrays are not supported")
        shape = tuple(int(dimension) for dimension in value["shape"])
        if any(dimension < 0 for dimension in shape):
            raise ProtocolError("array dimensions cannot be negative")
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(data) != expected_size:
            raise ProtocolError(
                f"array payload has {len(data)} bytes; shape and dtype require {expected_size}"
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    return {key: _decode(item) for key, item in value.items()}
